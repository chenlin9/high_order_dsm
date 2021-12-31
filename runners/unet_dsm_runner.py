import os
import shutil
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as utils
from model.models import *
from model.cond_refinenet_dilated import *
from functions.loss import *
from functions.utils import *

import dataset
import itertools
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from model.ema import EMAHelper

# Training s1 with dsm, s2 with second order sm together
class UnetDSMRunner(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        os.makedirs(os.path.join(self.args.log, 'images'), exist_ok=True)

        if self.config.loss == "dsm":
            self.loss_op1 = dsm
        elif self.config.loss == "dsm_vr":
            self.loss_op1 = dsm_vr
        else:
            raise Exception('{} loss function not in {dsm, dsm_vr}'.format(self.config.loss))


    def train(self):
        obs = (1, 28, 28) if 'MNIST' in self.config.dataset else (3, 32, 32)
        channel = self.channels = obs[0]
        img = self.image_size = obs[1]

        dim = channel * img * img
        train_loader, test_loader = dataset.get_dataset(self.config)

        # load s1 model
        s1 = CondRefineNetDilated(self.config).to(self.config.device)
        s1 = torch.nn.DataParallel(s1)

        optimizer = optim.Adam(itertools.chain(s1.parameters()), lr=self.config.lr)

        ckpt_path = os.path.join(self.args.log, 'ckpt')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(s1)

        if self.args.resume_training:
            state_dict = torch.load(os.path.join(ckpt_path, 'checkpoint.pth'), map_location=self.config.device)
            s1.load_state_dict(state_dict[0])
            optimizer.load_state_dict(state_dict[1])
            # scheduler.load_state_dict(state_dict[2])
            if self.config.model.ema:
                ema_helper.load_state_dict(states[2])
            print('model parameters loaded', flush=True)

        tb_path = os.path.join(self.args.log, 'tensorboard')
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        os.makedirs(tb_path)
        tb_logger = SummaryWriter(log_dir=tb_path)

        print('starting training', flush=True)
        writes = 0
        self.sigma = self.config.gaussian_sigma
        for epoch in range(self.config.max_epochs):
            train_loss1 = 0.
            s1.train()
            for batch_idx, (X, _) in enumerate(train_loader):
                X = X.cuda(non_blocking=True)
                if channel == 3:
                    X = X * 2. - 1. # rescale to [-1, 1] for cifar10

                X = X.view(X.shape[0], -1)
                loss = self.loss_op1(s1, X, self.sigma) # dsm training s1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(s1)

                train_loss1 += loss.item()
                if (batch_idx + 1) % self.config.print_every == 0:
                    deno = self.config.print_every
                    train_loss1 = train_loss1 / deno

                    print('epoch: {}, batch: {}, loss1 : {:.4f}'.format(epoch, batch_idx, train_loss1), flush=True)
                    tb_logger.add_scalar('loss1', train_loss1, global_step=writes)
                    train_loss1 = 0.
                    writes += 1
               
            # Evaluation
            with torch.no_grad():
                if self.config.model.ema:
                    test_s1 = ema_helper.ema_copy(s1)
                else:
                    test_s1 = s1

                test_s1.eval()
                test_loss1 = 0.
                for batch_idx, (X, _) in enumerate(test_loader):
                    X = X.cuda(non_blocking=True)
                    if channel == 3:
                        X = X * 2. - 1.  # rescale to [-1, 1] for cifar10

                    X = X.view(X.shape[0], -1)
                    loss = self.loss_op1(test_s1, X, self.sigma)
                    test_loss1 += loss.item()

                deno = batch_idx + 1.
                test_loss1 = test_loss1 / deno

                print('epoch: %s, test loss1 : %s' % (epoch, test_loss1), flush=True)
                tb_logger.add_scalar('test_loss1', test_loss1, global_step=writes)

            if (epoch + 1) % self.config.save_interval == 0:
                state_dict = [
                    s1.state_dict(),
                    optimizer.state_dict(),
                ]
                if self.config.model.ema:
                    state_dict.append(ema_helper.state_dict())

                torch.save(state_dict, os.path.join(ckpt_path, 'checkpoint_{}.pth'.format(epoch)))
                torch.save(state_dict, os.path.join(ckpt_path, 'checkpoint.pth'))
                print('Computing covariance...', flush=True)

                with torch.no_grad():
                    try:
                        test_X, _ = next(test_iter)
                    except:
                        test_iter = iter(test_loader)
                        test_X, _ = next(test_iter)

                    test_X = test_X[:5].to(self.config.device)
                    if channel == 3:
                        test_X = test_X * 2. - 1.  # rescale to [-1, 1] for cifar10

                    perturb_X = test_X + torch.randn_like(test_X) * self.sigma
                    perturb_X = perturb_X.view(perturb_X.shape[0], -1)
                    score1 = test_s1(perturb_X)
                    denoised = perturb_X + self.sigma ** 2 * score1
                    denoised = denoised.reshape(-1, channel, img, img)
                    denoised = torch.cat([perturb_X.reshape(perturb_X.shape[0], channel, img, img), test_X, denoised], dim=0)

                    if channel == 3:
                        denoised = denoised * 0.5 + 0.5

                    save_image(denoised,
                               os.path.join(self.args.log, 'images', 'denoised_{}.png'.format(epoch)), nrow=test_X.shape[0],
                               pad_value=1.)



    def langevin_dynamics_sampling(self, score1, dimension, num=100, lr=1e-2, step=1000):
        with torch.no_grad():
            samples = torch.randn(num, dimension).to(self.config.device)
            eps = lr
            for i in tqdm(range(step)):
                element_score = score1(samples).reshape(num, -1)  # without annealing
                samples = samples + eps * element_score * 0.5 + torch.randn_like(samples) * np.sqrt(eps)
            return samples

    def test(self):
        obs = (1, 28, 28) if 'MNIST' in self.config.dataset else (3, 32, 32)
        channel = self.channels = obs[0]
        img = self.image_size = obs[1]
        dim = channel * img * img

        train_loader, test_loader = dataset.get_dataset(self.config)
        self.sigma = self.config.gaussian_sigma

        if self.args.rank <= 0:
            self.args.rank = 10

        model = CondRefineNetDilated(self.config)
        model = model.to(self.config.device)
        model = torch.nn.DataParallel(model)

        ckpt_path = os.path.join(self.args.log, 'ckpt')
        state_dict = torch.load(os.path.join(ckpt_path, 'checkpoint.pth'), map_location=self.config.device)
        model.load_state_dict(state_dict[0])
        model.eval()
        os.makedirs(os.path.join(self.args.log, 'eval_images'), exist_ok=True)
        print('Computing covriance...', flush=True)

        # load s1 model
        s1 = CondRefineNetDilated(self.config).to(self.config.device)
        s1 = torch.nn.DataParallel(s1)
        s1.load_state_dict(state_dict[1])
        s1.eval()


        with torch.no_grad():
            model.eval()
            try:
                test_X, _ = next(test_iter)
            except:
                test_iter = iter(test_loader)
                test_X, _ = next(test_iter)

            test_X = test_X[:5].to(self.config.device)

            print("start sampling")
            num = 16
            sample_lr = 1e-3
            step = 200
            samples = self.langevin_dynamics_sampling(s1, dim, num=num, lr=sample_lr, step=step)
            samples = samples.reshape(samples.shape[0], channel, img, img)
            save_image(samples,
                       os.path.join(self.args.log, 'eval_images', 'naive_samples.png'), nrow=int(num**0.5), pad_value=1.)
            denoised_samples = samples + self.sigma ** 2 * s1(samples).reshape(samples.shape[0], channel, img, img)
            save_image(denoised_samples,
                       os.path.join(self.args.log, 'eval_images', 'denoised_naive_samples.png'), nrow=int(num ** 0.5),
                       pad_value=1.)

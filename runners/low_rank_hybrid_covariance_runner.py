from torch.utils.tensorboard import SummaryWriter
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
from torchvision.utils import make_grid, save_image


# Training s1 with dsm, s2 with second order sm together
class LowRankHybridRunner(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        os.makedirs(os.path.join(self.args.log, 'images'), exist_ok=True)

        if self.config.loss == "dsm":
            self.loss_op1 = dsm_vr
        elif self.config.loss == "dsm_vr":
            self.loss_op1 = dsm_vr
        else:
            raise Exception('{} loss function not in {dsm, dsm_vr}'.format(self.config.loss))

        if self.config.loss == "dsm":
            self.loss_op2 = hosm_low_rank
        elif self.config.loss == "dsm_vr":
            self.loss_op2 = hosm_plus_vr_low_rank
        else:
            raise Exception('{} loss function not in {dsm, dsm_vr}'.format(self.config.loss))


    def train(self):
        obs = (1, 28, 28) if 'MNIST' in self.config.dataset else (3, 32, 32)
        channel = self.channels = obs[0]
        img = self.image_size = obs[1]

        dim = channel * img * img
        train_loader, test_loader = dataset.get_dataset(self.config)

        if self.args.rank <= 0:
            self.args.rank = 10
        model = LowRankS2(self.config) # check the trainable parameters, make sure both models
        model = model.to(self.config.device)

        # load s1 model
        s1 = CondRefineNetDilated(self.config).to(self.config.device)
        s1 = torch.nn.DataParallel(s1)

        optimizer = optim.Adam(itertools.chain(model.low_rank_model.parameters(), model.diagonal.parameters(), s1.parameters()), lr=self.config.lr)
        model = torch.nn.DataParallel(model)

        ckpt_path = os.path.join(self.args.log, 'ckpt')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        if self.args.resume_training:
            state_dict = torch.load(os.path.join(ckpt_path, 'checkpoint.pth'), map_location=self.config.device)
            model.load_state_dict(state_dict[0])
            optimizer.load_state_dict(state_dict[1])
            scheduler.load_state_dict(state_dict[2])
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
            train_loss2 = 0.
            model.train()
            s1.train()
            for batch_idx, (X, _) in enumerate(train_loader):
                X = X.cuda(non_blocking=True)
                X = X.view(X.shape[0], -1)

                loss1 = 10. * self.loss_op1(s1, X, self.sigma) / (self.sigma ** 2)  # dsm training s1
                loss2 = self.loss_op2(s1, model, X, self.sigma) # second order sm loss
                loss = loss1 + loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
                if (batch_idx + 1) % self.config.print_every == 0:
                    deno = self.config.print_every
                    train_loss1 = train_loss1 / deno
                    train_loss2 = train_loss2 / deno

                    print('epoch: {}, batch: {}, loss1 : {:.4f}, loss2 : {:.4f}'.format(epoch, batch_idx, train_loss1, train_loss2), flush=True)
                    tb_logger.add_scalar('loss1', train_loss1, global_step=writes)
                    tb_logger.add_scalar('loss2', train_loss2, global_step=writes)
                    train_loss1 = 0.
                    train_loss2 = 0.
                    writes += 1

            model.eval()
            s1.eval()
            test_loss1 = 0.
            test_loss2 = 0.
            with torch.no_grad():
                for batch_idx, (X, _) in enumerate(test_loader):
                    X = X.cuda(non_blocking=True)
                    X = X.view(X.shape[0], -1)
                    loss1 = self.loss_op1(s1, X, self.sigma)
                    loss2 = self.loss_op2(s1, model, X, self.sigma)
                    test_loss1 += loss1.item()
                    test_loss2 += loss2.item()
                    break # reduce test iterations

                deno = batch_idx + 1.
                test_loss1 = test_loss1 / deno
                test_loss2 = test_loss2 / deno

                print('epoch: %s, test loss1 : %s, test loss2 : %s' % (epoch, test_loss1, test_loss2), flush=True)
                tb_logger.add_scalar('test_loss1', test_loss1, global_step=writes)
                tb_logger.add_scalar('test_loss2', test_loss2, global_step=writes)

            if (epoch + 1) % self.config.save_interval == 0:
                state_dict = [
                    model.state_dict(),
                    s1.state_dict(),
                    optimizer.state_dict(),
                ]
                torch.save(state_dict, os.path.join(ckpt_path, 'checkpoint_{}.pth'.format(epoch)))

            if (epoch + 1) % 1 == 0:
                state_dict = [
                    model.state_dict(),
                    s1.state_dict(),
                    optimizer.state_dict(),
                ]
                torch.save(state_dict, os.path.join(ckpt_path, 'checkpoint.pth'))
                print('Computing covariance...', flush=True)

                with torch.no_grad():
                    try:
                        test_X, _ = next(test_iter)
                    except:
                        test_iter = iter(test_loader)
                        test_X, _ = next(test_iter)

                    test_X = test_X[:5].to(self.config.device)
                    perturb_X = test_X + torch.randn_like(test_X) * self.sigma
                    perturb_X = perturb_X.view(perturb_X.shape[0], -1)
                    score1 = s1(perturb_X)
                    denoised = perturb_X + self.sigma ** 2 * score1
                    denoised = denoised.reshape(-1, channel, img, img)
                    denoised = torch.cat([perturb_X.reshape(perturb_X.shape[0], channel, img, img), test_X, denoised], dim=0)
                    save_image(denoised,
                               os.path.join(self.args.log, 'images', 'denoised_{}.png'.format(epoch)), nrow=test_X.shape[0],
                               pad_value=1.)

                    score_2 = model(perturb_X).detach().reshape(perturb_X.shape[0], dim, dim) #currently only support one channel input

                    with torch.no_grad():
                        covariance = score_2 + torch.eye(self.image_size ** 2, device=self.config.device).reshape(1, self.image_size ** 2, self.image_size ** 2) / (self.sigma ** 2)

                        cov_concats = []
                        for t in range(5):
                            concat = covariance[t].max(dim=0)[0].view(channel, img, img)  # covariance: batch, img * img, img * img
                            cov_concats.append(concat)

                        cov_concats = torch.stack(cov_concats, dim=0)
                        test_X = torch.cat([test_X, cov_concats], dim=0)
                        save_image(test_X, os.path.join(self.args.log, 'images', 'source_{}.png'.format(epoch)), nrow=5)

    def langevin_dynamics_sampling(self, score1, dimension, num=100, lr=1e-2, step=1000):
        with torch.no_grad():
            samples = torch.randn(num, dimension).to(self.config.device)
            eps = lr
            for i in tqdm(range(step)):
                element_score = score1(samples).reshape(num, -1)  # without annealing
                samples = samples + eps * element_score * 0.5 + torch.randn_like(samples) * np.sqrt(eps)
            return samples

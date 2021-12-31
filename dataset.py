import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, ImageFolder


def get_dataset(config):
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    rescaling = lambda x: (x - .5) * 2.
    # ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
    ds_transforms = transforms.Compose([transforms.ToTensor()]) # removed rescaling

    if config.dataset == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(config.data_dir, download=True,
                           train=True, transform=ds_transforms),
            batch_size=config.batch_size,
            shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(datasets.MNIST(config.data_dir, train=False, download=True,
                                                                 transform=ds_transforms), batch_size=config.batch_size,
                                                  shuffle=False, **kwargs)

    elif config.dataset == 'FashionMNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(config.data_dir, download=True,
                           train=True, transform=ds_transforms),
            batch_size=config.batch_size,
            shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(config.data_dir, train=False, download=True,
                                                                 transform=ds_transforms), batch_size=config.batch_size,
                                                  shuffle=False, **kwargs)


    elif 'CIFAR10' in config.dataset:

        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(config.data_dir, train=True,
                                                                    download=True, transform=ds_transforms),
                                                   batch_size=config.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(config.data_dir, train=False, download=True,
                                                                   transform=ds_transforms),
                                                  batch_size=config.batch_size,
                                                  shuffle=False, **kwargs)

    elif "celeba" in config.dataset:
        dataset = ImageFolder(
            root=config.data_dir,
            transform=transforms.Compose([
                transforms.CenterCrop(140),
                transforms.Resize(32),
                transforms.ToTensor(),
                rescaling
            ]))
        num_items = len(dataset)  # 202599
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2020)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.7)], indices[
                                                                      int(num_items * 0.7):int(num_items * 0.8)]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

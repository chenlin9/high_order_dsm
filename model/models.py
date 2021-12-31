import torch
import torch.nn as nn


class low_rank_s2(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()

        self.dim = dim # 784
        self.rank = rank
        self.main = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, self.dim * rank)
        )
        self.diag = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, self.dim)
        )

    def forward(self, X):
        X = X.reshape(X.shape[0], -1)
        cov_factor = self.main(X).reshape(X.shape[0], self.dim, self.rank)
        cov_product = torch.matmul(cov_factor, cov_factor.permute(0, 2, 1)) # batch, dim, dim
        cov_diag = self.diag(X) # batch, dim
        cov_diag = torch.diag_embed(cov_diag, offset=0, dim1=-2, dim2=-1) # batch, dim, dim
        h = cov_product + cov_diag
        return h


class diagonal_s2(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()

        self.dim = dim # 784
        self.rank = rank
        self.diag = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, self.dim)
        )

    def forward(self, X):
        # import pdb
        # pdb.set_trace()
        X = X.reshape(X.shape[0], -1)
        cov_diag = self.diag(X) # batch, dim
        cov_diag = torch.diag_embed(cov_diag, offset=0, dim1=-2, dim2=-1) # batch, dim, dim
        h = cov_diag
        return h


class MLPScore(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.main = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, self.dim)
        )

    def forward(self, X):
        h = self.main(X)
        return h


class ConvAutoencoder(nn.Module):
    def __init__(self, image_size=28):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.Softplus(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.Softplus(),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1)
        )
        self.image_size = image_size

    def forward(self, x):
        x = x.view(x.shape[0], 1, self.image_size, self.image_size)
        x = self.main(x)
        x = x.view(x.shape[0], -1)

        return x


class CIFAR10ConvAutoencoder(nn.Module):
    def __init__(self, image_size=32):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.Softplus(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.Softplus(),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(64, 3, 5, stride=2, padding=2, output_padding=1)
        )
        self.image_size = image_size

    def forward(self, x):
        x = x.view(x.shape[0], 3, self.image_size, self.image_size)
        x = self.main(x)
        x = x.view(x.shape[0], -1)

        return x

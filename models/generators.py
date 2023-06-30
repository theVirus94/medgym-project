# Different Generators for different classes of subsets

import torch
from torch import nn, Tensor
import numpy as np

class LungOpacityGenerator(nn.Module):
    def __init__(self,n_features=100):
        super(LungOpacityGenerator, self).__init__()
        self.n_features = n_features
        self.n_out = (3, 299, 299)

        nc, nz = 3, n_features

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(n_features, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )

        # Try to set the problem.

        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )

        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )

        self.hidden4 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )


        self.out = nn.Sequential(
            nn.ConvTranspose2d(8, 3, 4, 2, 1, bias=False),
            nn.Upsample(size=(299, 299)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.out(x)
        return x
class COVID19Generator(nn.Module):
    def __init__(self, n_features=100):
        super(COVID19Generator, self).__init__()
        self.n_features = n_features
        self.n_out = (3, 299, 299)

        nc, nz, ngf = 3, n_features, 64

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),
        )

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),
        )

        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),
        )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
class PneumoniaGenerator(nn.Module):
    def __init__(self, n_features=100):
        super(PneumoniaGenerator, self).__init__()
        self.n_features = n_features
        self.n_out = (3, 299, 299)

        nc, nz, ngf = 3, n_features, 64

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),
        )

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),
        )

        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),
        )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

# Generator for the combined dataset (Normal + Lung Opacity)
class LungOpacity_NormalGenerator(nn.Module):
    def __init__(self, n_features=100):
        super(LungOpacity_NormalGenerator, self).__init__()
        self.n_features = n_features
        self.num_classes = 2
        self.img_size = 299

        self.label_emb = nn.Embedding(self.num_classes, self.n_features)

        self.embed_layer = nn.Sequential(
            nn.ConvTranspose2d(self.n_features + self.n_features, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(0.2),
        )

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(0.2),
        )

        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(0.2),
        )

        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(0.2),
        )

        self.hidden4 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(8, 3, 4, 2, 1, bias=False),
            nn.Upsample(size=(299, 299)),
            nn.Tanh()
        )

    def forward(self, x, labels):
        labels = self.label_emb(labels)
        labels = labels.view(labels.size(0), self.n_features, 1, 1)

        x = torch.cat((x, labels), dim=1)

        x = self.embed_layer(x)
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.out(x)
        return x
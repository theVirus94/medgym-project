
import torch
from torch import nn
import numpy as np

class LungOpacityDiscriminator(nn.Module):
    def __init__(self):
        super(LungOpacityDiscriminator, self).__init__()
        self.n_features = (3, 299, 299)
        nc, ndf = 3, 64

        self.input_layer = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x.view(x.size(0), 1, 1, 1)
class COVID19Discriminator(nn.Module):
    def __init__(self):
        super(COVID19Discriminator, self).__init__()
        self.n_features = (3, 299, 299)
        nc, ndf = 3, 64

        self.input_layer = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Dropout2d(p=0.2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("x before input layer", x.size())
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
class PneumoniaDiscriminator(nn.Module):
    def __init__(self):
        super(PneumoniaDiscriminator, self).__init__()
        self.n_features = (3, 299, 299)
        nc, ndf = 3, 64

        self.input_layer = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Dropout2d(p=0.2),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print("x before input layer", x.size())
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


# Discriminator for the combined dataset (Normal + Lung Opacity)
class LungOpacity_NormalDiscriminator(nn.Module):
    def __init__(self):
        super(LungOpacity_NormalDiscriminator, self).__init__()
        self.image_size = 299
        self.num_classes = 2
        self.image_channels = 3

        self.embedding_layer = nn.Embedding(self.num_classes, self.image_size * self.image_size)

        self.embed_layer = nn.Sequential(
            nn.ConvTranspose2d(self.image_channels + 1, 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout2d(0.2)
        )

        self.input_layer = nn.Sequential(
            nn.Conv2d(8, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        )

        self.hidden3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        )

        self.out = nn.Sequential(
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Dropout2d(p=0.2),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # print("x before input layer", x.size())
        labels = self.embedding_layer(labels)
        labels = labels.view(labels.size(0), 1, self.image_size, self.image_size)

        x = torch.cat([x, labels], 1)

        x = self.embed_layer(x)

        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)

        x = self.out(x)
        return x
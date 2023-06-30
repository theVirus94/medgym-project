# Enumerator for different test configurations

from enum import Enum

from experiment import Experiment
from torch import nn, optim
from models.generators import LungOpacityGenerator, COVID19Generator, PneumoniaGenerator, LungOpacity_NormalGenerator
from models.discriminators import LungOpacityDiscriminator, COVID19Discriminator, PneumoniaDiscriminator, LungOpacity_NormalDiscriminator

class ExperimentEnums(Enum):
    """
    Enumerator for the experiments, right now we have onlu one which is Lung Opacity.
    """

    # LungOpacityModel = {
    #     "noise_emb_sz": 100,
    #
    #     "generator": LungOpacityGenerator,
    #     "discriminator": LungOpacityDiscriminator,
    #
    #     "dataset": "Lung_Opacity",
    #
    #     "target_img_width": 299,
    #     "target_img_length": 299,
    #     "batch_size": 32,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "gen_lr": 0.0002,
    #     "disc_lr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # COVID19Model = {
    #     "noise_emb_sz": 100,
    #
    #     "generator": COVID19Generator,
    #     "discriminator": COVID19Discriminator,
    #
    #     "dataset": "COVID19",
    #
    #     "target_img_width": 299,
    #     "target_img_length": 299,
    #     "batch_size": 16,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "gen_lr": 0.0002,
    #     "disc_lr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }

    # Pneumonia = {
    #     "noise_emb_sz": 100,
    #
    #     "generator": PneumoniaGenerator,
    #     "discriminator": PneumoniaDiscriminator,
    #
    #     "dataset": "Pneumonia",
    #
    #     "target_img_width": 299,
    #     "target_img_length": 299,
    #     "batch_size": 16,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "percentage": 1,
    #     "gen_lr": 0.0002,
    #     "disc_lr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }

    LungOpacity_NormalModel = {
        "noise_emb_sz": 100,

        "generator": LungOpacity_NormalGenerator,
        "discriminator": LungOpacity_NormalDiscriminator,

        "dataset": "LungOpacity_Normal",

        "target_img_width": 299,
        "target_img_length": 299,
        "batch_size": 64,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "percentage": 1,
        "gen_lr": 0.0001,
        "disc_lr": 0.0001,
        "loss": nn.BCELoss(),
        "epochs": 50
    }

    def __str__(self):
        return self.value

experimentsAll = [Experiment(expType=i) for i in ExperimentEnums]
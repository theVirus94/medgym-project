from torch.autograd import Variable
from torch import nn
import torch
import time
import random
import string
import pandas as pd
import numpy as np
from utils.vector_utils import values_target, weights_init, vectors_to_images
from get_data import get_loader
from logger import Logger
from utils.utils import label_smoothing
import matplotlib.pyplot as plt
from matplotlib import cm

class Experiment:

    def __init__(self, expType):

        self.name = expType.name
        self.type = expType.value

        self.noise_emb_sz = self.type["noise_emb_sz"]

        self.target_image_w = self.type["target_img_width"]
        self.target_image_h = self.type["target_img_length"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.generator = self.type["generator"](n_features=self.noise_emb_sz).to(self.device)
        self.discriminator = self.type["discriminator"]().to(self.device)

        self.g_optim = self.type["g_optim"](self.generator.parameters(),
                                            lr=self.type["gen_lr"], betas=(0.5, 0.99))
        self.d_optim = self.type["d_optim"](self.discriminator.parameters(),
                                            lr=self.type["disc_lr"], betas=(0.5, 0.99))
        self.loss = self.type["loss"]
        self.epochs = self.type["epochs"]
        self.cuda = True if torch.cuda.is_available() else False
        self.real_label = 0.9
        self.fake_label = 0.1
        self.samples = 16

    def run(self, logging_frequency=4):
        """
        Runs the experiments
        """
        start_time = time.time()

        logger = Logger(self.name, self.type["dataset"])

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        loader = get_loader(self.type["dataset"], self.type["batch_size"], self.type["percentage"], self.target_image_w,
                            self.target_image_h)
        num_batches = len(loader)

        if self.cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.loss = self.loss.cuda()

        G_losses = []
        D_losses = []
        best_g_error = 10000
        best_d_error = 10000
        num_classes = 2

        # TODO Training Loop
        for epoch in range(1, self.epochs + 1):
            for n_batch, sample in enumerate(loader):
                torch.cuda.empty_cache()

                N = len(sample)

                batch_images = torch.stack([sample[i][0] for i in range(0,N)], dim=0)
                label_batch = torch.stack([sample[i][1] for i in range(0, N)], dim=0)

                batch_images = batch_images.to(self.device)
                label_batch = label_batch.to(self.device)

                noise = torch.randn(N, self.noise_emb_sz, 1, 1).to(self.device)
                noise = (noise + 1) / 2

                # Also add some noise to the real data to prevent discriminator overfitting problem
                real_noise = torch.randn_like(batch_images)
                batch_images = batch_images + real_noise

                # Apply label smoothing to the labels
                label_batch = label_smoothing(label_batch).type(torch.LongTensor).to(self.device)

                noise = noise.to(self.device)

                fake_images = self.generator(noise, label_batch).detach()

                # Creating fake labels for fake images.
                # TODO: Can we create these labels based on a criterion?
                fake_labels = torch.randint(low=0, high=num_classes, size=(N,)).to(self.device)

                # Train D
                d_error, d_pred_real, d_pred_fake = self.train_discriminator(real_data=batch_images,
                                                                             fake_data=fake_images,
                                                                             real_labels=label_batch,
                                                                             fake_labels=fake_labels)

                # Create new noise
                noise = torch.randn(N, self.noise_emb_sz, 1, 1)
                noise = noise.to(self.device)

                # Create fake data with this noise
                fake_labels = torch.randint(low=0, high=num_classes, size=(N,)).to(self.device)
                fake_data = self.generator(noise, fake_labels)

                # Train G
                g_error = self.train_generator(fake_data=fake_data, fake_labels=fake_labels)

                if g_error <= best_g_error:
                    logger.save_model(model=self.generator, name="generator", epoch=epoch, loss=g_error)
                    best_g_error = g_error
                if d_error <= best_d_error:
                    logger.save_model(model=self.discriminator, name="discriminator", epoch=epoch, loss=d_error)
                    best_d_error = d_error

                # Save Losses for plotting later
                G_losses.append(g_error.item())
                D_losses.append(d_error.item())

                logger.log(d_error, g_error, epoch, n_batch, num_batches)

                # Display status Logs
                if n_batch % (num_batches // logging_frequency) == 0:
                    logger.display_status(
                        epoch, self.epochs, n_batch, num_batches,
                        d_error, g_error, d_pred_real, d_pred_fake
                        )

        logger.save_errors(g_loss=G_losses, d_loss=D_losses)
        timeTaken = time.time() - start_time

        test_sample_size = 16
        test_noise = torch.randn(test_sample_size, self.noise_emb_sz, 1, 1)
        test_noise = (test_noise + 1) / 2

        test_noise = test_noise.to(self.device)
        test_labels = torch.randint(low=0, high=num_classes, size=(test_sample_size,)).to(self.device)
        test_images = self.generator(test_noise, test_labels)

        test_images = vectors_to_images(test_images, self.target_image_w, self.target_image_h).cpu().data

        logger.log_images(test_images, self.epochs + 1, 0, num_batches)
        logger.save_scores(timeTaken, 0)
        return


    def train_generator(self, fake_data, fake_labels):
        """
        This function performs one iteration of training the generator
        :param
        """
        N = fake_data.size(0)

        self.g_optim.zero_grad()
        # Sample the noise and generate fake_data
        prediction = self.discriminator(fake_data, fake_labels).view(-1)

        error = self.loss(prediction, values_target(size=(prediction.size(0),), value=self.real_label, cuda=self.cuda))

        error.backward()

        # clip gradients to avoid exploding gradient problem
        nn.utils.clip_grad_norm_(self.generator.parameters(), 10)

        # update parameters
        self.g_optim.step()

        # Return error
        return error

    def train_discriminator(self, real_data, fake_data, real_labels, fake_labels):
        """
        This function performs one iteration of training the discriminator
        :param
        """
        N = real_data.size(0)
        real_data = real_data.float()

        self.d_optim.zero_grad()

        # Train on real data
        prediction_real = self.discriminator(real_data, real_labels).view(-1)
        error_real = self.loss(prediction_real, values_target(size=(prediction_real.size(0),), value=self.real_label, cuda=self.cuda))

        # Train on fake data
        prediction_fake = self.discriminator(fake_data, fake_labels).view(-1)  # .to(torch.float16)
        error_fake = self.loss(prediction_fake, values_target(size=(prediction_fake.size(0),), value=self.fake_label, cuda=self.cuda))

        error = error_real + error_fake
        error.backward()

        self.d_optim.step()

        # Return error and predictions for real and fake inputs
        return (error_real + error_fake) / 2, prediction_real, prediction_fake

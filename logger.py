import torch
import errno
import os
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils


class Logger:
    """
    Logs information during the experiment
    """

    def __init__(self, experiment_name, datasetName):
        """
        Standard init
        :param experiment_name: name of the experiment enum
        :type experiment_name: str
        :param datasetName: name of the dataset
        :type datasetName: str
        """
        self.model_name = experiment_name
        self.data_subdir = f'./results/{datasetName}/{experiment_name}'
        Logger._make_dir(self.data_subdir)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.data_subdir, write_to_disk=False)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/D_error'.format(self.data_subdir), d_error, step)
        self.writer.add_scalar(
            '{}/G_error'.format(self.data_subdir), g_error, step)

    def save_errors(self, g_loss, d_loss):
        np.save(self.data_subdir + "/g_loss.npy", np.array(g_loss))
        np.save(self.data_subdir + "/d_loss.npy", np.array(d_loss))

        plt.plot(g_loss, color="blue", label="generator")
        plt.plot(d_loss, color="orange", label="discriminator")
        plt.legend()
        plt.savefig(self.data_subdir + "/plotLoss.png")

    def Generator_per_epoch(self, fake_data, epoch):
        from PIL import Image
        path_output = f'{os.getcwd()}/Generated_PerEpoch'
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        # Added for device incompatibility - Caner
        fake_data = fake_data.detach().cpu().numpy()

        fake_sample = np.transpose(fake_data, (1, 2, 0))
        fake_sample = ((fake_sample / 2) + 0.5) * 255
        fake_sample = fake_sample.astype(np.uint8)
        fake_sample_image = Image.fromarray(fake_sample)
        fake_sample_image.save(f'{path_output}/{epoch}.jpg')

    def log_images(self, images, epoch, n_batch, num_batches, i_format='NCHW', normalize=True):
        """ input images are expected in format (NCHW) """
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)

        if i_format == 'NHWC':
            images = images.transpose(1, 3)

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.data_subdir, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, epoch, n_batch):
        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.title("Test")
        plt.axis('off')
        fig.savefig('{}/epoch_{}_batch_{}.png'.format(self.data_subdir, epoch, n_batch))
        plt.close()

    @staticmethod
    def display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, num_batches)
        )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, generator):
        torch.save(generator.state_dict(), f'{self.data_subdir}/generator.pt')

    def save_model(self, model, name="generator"):
        torch.save(model.state_dict(), f'{self.data_subdir}/{name}.pt')

    def save_model(self, model, name, epoch, loss, optimizer=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'{self.data_subdir}/{name}.pt')

    def savefig(self, fig, filename):
        fig.savefig(f'{self.data_subdir}/{filename}.png')

    def close(self):
        self.writer.close()

    # Private Functions
    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def save_scores(self, time, fid):
        with open(f'{self.data_subdir}/results.txt', 'w') as file:
            file.write(f'time taken: {round(time, 4)}\n')
            file.write(f'fid score: {round(fid, 4)}')
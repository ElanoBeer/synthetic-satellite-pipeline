import argparse
import os
import numpy as np
from PIL import Image
import math
import sys
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

# Create directories for outputs
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# Command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
    parser.add_argument("--log_interval", type=int, default=10, help="interval between log outputs")
    return parser.parse_args()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 1024),  # Increased capacity for larger images
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class ImageDataset(Dataset):
    def __init__(self, roots, transform=None):
        self.files = []

        # Handle both single directory (string) and multiple directories (list)
        if isinstance(roots, str):
            roots = [roots]

        # Collect files from all directories
        for root in roots:
            files_in_dir = [os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.files.extend(files_in_dir)

        self.transform = transform
        print(f"Total images found: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Get appropriate tensor type based on device
    Tensor = torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor

    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(opt, generator, discriminator, dataloader, compute_gradient_penalty):
    """Main training function"""
    # For tracking and visualizing losses
    d_losses = []
    g_losses = []
    epoch_d_losses = []
    epoch_g_losses = []

    start_time = time.time()
    total_batches = len(dataloader)
    batches_done = 0

    # Setup device and tensor type
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if cuda else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        epoch_start_time = time.time()
        epoch_d_loss = 0
        epoch_g_loss = 0
        generator_updates = 0

        progress_bar = tqdm(enumerate(dataloader), total=total_batches,
                            desc=f"Epoch {epoch + 1}/{opt.n_epochs}")

        for i, imgs in progress_bar:
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs.detach())
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, device)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            # Track discriminator loss
            current_d_loss = d_loss.item()
            d_losses.append(current_d_loss)
            epoch_d_loss += current_d_loss

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                # Track generator loss
                current_g_loss = g_loss.item()
                g_losses.append(current_g_loss)
                epoch_g_loss += current_g_loss
                generator_updates += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'D_loss': f'{current_d_loss:.4f}',
                    'G_loss': f'{current_g_loss:.4f}',
                    'GP': f'{gradient_penalty.item():.4f}'
                })

                # Log detailed info at intervals
                if batches_done % opt.log_interval == 0:
                    elapsed = time.time() - start_time
                    print(f"\nBatch {i + 1}/{total_batches} | "
                          f"D loss: {current_d_loss:.4f} | "
                          f"G loss: {current_g_loss:.4f} | "
                          f"Time: {elapsed:.2f}s")

                # Save sample images
                if batches_done % opt.sample_interval == 0:
                    save_image(fake_imgs.data[:25], f"images/epoch_{epoch + 1}_batch_{batches_done}.png",
                               nrow=5, normalize=True)

                batches_done += 1

        # Calculate average epoch losses
        avg_d_loss = epoch_d_loss / total_batches
        avg_g_loss = epoch_g_loss / generator_updates if generator_updates > 0 else 0
        epoch_d_losses.append(avg_d_loss)
        epoch_g_losses.append(avg_g_loss)

        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        print(f"\n[Epoch {epoch + 1}/{opt.n_epochs}] "
              f"Avg D loss: {avg_d_loss:.4f} | "
              f"Avg G loss: {avg_g_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")

        # Save model checkpoints
        if (epoch + 1) % opt.checkpoint_interval == 0:
            torch.save(generator.state_dict(), f"models/generator_epoch_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch + 1}.pth")
            print(f"Checkpoint saved at epoch {epoch + 1}")

            # Create and save intermediate loss plots
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epoch_d_losses, 'b-', label='Discriminator')
            plt.plot(epoch_g_losses, 'r-', label='Generator')
            plt.title('Training Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(d_losses[-100:], 'b-', label='Discriminator', alpha=0.5)
            plt.plot(g_losses[-100:], 'r-', label='Generator', alpha=0.5)
            plt.title('Recent Training Loss per Batch')
            plt.xlabel('Batch (last 100)')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"plots/losses_epoch_{epoch + 1}.png", dpi=300)
            plt.close()

    # Return losses for visualization
    return generator, discriminator, epoch_d_losses, epoch_g_losses, d_losses, g_losses


def save_final_results(generator, discriminator, epoch_d_losses, epoch_g_losses, d_losses, g_losses):
    # Save final models
    torch.save(generator.state_dict(), "models/generator_final.pth")
    torch.save(discriminator.state_dict(), "models/discriminator_final.pth")
    print("Final models saved")

    # Create final loss visualization
    plt.figure(figsize=(20, 10))

    # Plot epoch losses
    plt.subplot(2, 2, 1)
    plt.plot(epoch_d_losses, 'b-', label='Discriminator')
    plt.plot(epoch_g_losses, 'r-', label='Generator')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot batch losses
    plt.subplot(2, 2, 2)
    plt.plot(d_losses, 'b-', label='Discriminator', alpha=0.5)
    plt.plot(g_losses, 'r-', label='Generator', alpha=0.5)
    plt.title('Training Loss per Batch (All)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot recent batch losses (last 20% for clarity)
    recent_d = d_losses[-int(len(d_losses) * 0.2):] if d_losses else []
    recent_g = g_losses[-int(len(g_losses) * 0.2):] if g_losses else []

    plt.subplot(2, 2, 3)
    plt.plot(recent_d, 'b-', label='Discriminator')
    plt.plot(recent_g, 'r-', label='Generator')
    plt.title('Recent Training Loss (Last 20%)')
    plt.xlabel('Recent Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot moving average for smoothing
    window_size = 50
    if len(d_losses) > window_size:
        d_avg = np.convolve(d_losses, np.ones(window_size) / window_size, mode='valid')
        g_avg = np.convolve(g_losses, np.ones(window_size) / window_size, mode='valid')

        plt.subplot(2, 2, 4)
        plt.plot(d_avg, 'b-', label='Discriminator (Moving Avg)')
        plt.plot(g_avg, 'r-', label='Generator (Moving Avg)')
        plt.title(f'Moving Average Loss (Window={window_size})')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("plots/final_training_loss.png", dpi=300)
    plt.show()
    print("Final loss visualization saved as 'plots/final_training_loss.png'")


if __name__ == "__main__":
    # Parse command line arguments
    opt = parse_args()
    print(opt)

    # Set up image shape and device
    img_shape = (opt.channels, opt.img_size, opt.img_size)
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Using device: {device}")

    # Loss weight for gradient penalty
    lambda_gp = 10

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Set up dataset and dataloader
    # Transform
    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Image directories
    image_dirs = [
        r"E:/Datasets/masati-thesis/synthetic_images/0_original_images",
        r"E:/Datasets/masati-thesis/synthetic_images/1_object_insertion",
        r"E:/Datasets/masati-thesis/synthetic_images/2_augmentation",
        r"E:/Datasets/masati-thesis/synthetic_images/3_cloud_generation",
    ]

    # Create dataset and dataloader
    dataset = ImageDataset(image_dirs, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        drop_last=True
    )

    # Train model
    generator, discriminator, epoch_d_losses, epoch_g_losses, d_losses, g_losses = train(
        opt, generator, discriminator, dataloader, compute_gradient_penalty)

    # Save final results and visualize losses
    save_final_results(generator, discriminator, epoch_d_losses, epoch_g_losses, d_losses, g_losses)
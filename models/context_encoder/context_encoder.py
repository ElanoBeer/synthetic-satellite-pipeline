"""
Inpainting using Generative Adversarial Networks with object-based masking.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instructions on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
3. Prepare your annotation JSON file with bounding boxes
4. Run the script using command 'python3 context_encoder.py --annotation_file path/to/annotations.json'
"""

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=128, help="size of masked region")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument("--annotation_file", type=str, default='E:/Datasets/masati-thesis/agg_annotations.json', help="path to JSON file with bounding box annotations")
parser.add_argument("--dataset_path", type=str, default='E:/Datasets/masati-thesis/clones', help="path to dataset")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
# Use a consistent patch size of 128x128 (which is 1/4 of a 512x512 image)
patch_size = opt.mask_size  # This should match the mask_size used in the dataset
patch = (1, patch_size // 16, patch_size // 16)  # After 4 stride-2 layers in discriminator


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(channels=opt.channels)
discriminator = Discriminator(channels=opt.channels)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = DataLoader(
    ImageDataset(root=opt.dataset_path, transforms_=transforms_,
                 img_size=opt.img_size, mask_size=opt.mask_size,
                 mode="train", annotation_file=opt.annotation_file),
    batch_size=opt.batch_size,
    shuffle=True,
)
test_dataloader = DataLoader(
    ImageDataset(root=opt.dataset_path, transforms_=transforms_,
                 img_size=opt.img_size, mask_size=opt.mask_size,
                 mode="val", annotation_file=opt.annotation_file),
    batch_size=12,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def save_sample(batches_done):
    """Generate and save images during testing"""
    samples, masked_samples, masked_parts, coords = next(iter(test_dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    masked_parts = Variable(masked_parts.type(Tensor))

    # Generate inpainted masked parts
    gen_parts = generator(masked_samples)

    # Create filled samples by placing generated content back in the masked area
    filled_samples = masked_samples.clone()

    # For each sample, place the generated part at the right position
    for i in range(samples.size(0)):
        y1, x1 = coords[0][i].item(), coords[1][i].item()
        h, w = masked_parts[i].shape[1], masked_parts[i].shape[2]
        filled_samples[i, :, y1:y1+h, x1:x1+w] = gen_parts[i, :, :h, :w]  # Ensure dimensions match

    # Save sample
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, "images/%d.png" % batches_done, nrow=6, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, masked_imgs, masked_parts, coords) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        masked_parts = Variable(masked_parts.type(Tensor))


        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Instead of: gen_parts = generator(masked_imgs)
        gen_input = torch.zeros_like(masked_parts)
        for j in range(masked_imgs.size(0)):
            y1, x1 = coords[0][j].item(), coords[1][j].item()
            gen_input[j] = masked_imgs[j, :, y1:y1 + opt.mask_size, x1:x1 + opt.mask_size]

        # Generate a batch of images
        gen_parts = generator(gen_input)

        # # Generate a batch of images
        # gen_parts = generator(masked_imgs)

        #print(discriminator(gen_parts).shape,valid.shape, masked_parts.shape, masked_imgs.shape, gen_parts.shape)

        # Adversarial and pixelwise loss
        g_adv = adversarial_loss(discriminator(gen_parts), valid)
        g_pixel = pixelwise_loss(gen_parts, masked_parts)

        # Total loss (higher weight on pixel loss to focus on reconstruction quality)
        g_loss = 0.001 * g_adv + 0.999 * g_pixel

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(masked_parts), valid)
        fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
        )

        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)

    # Save model checkpoints after each epoch
    if epoch % 10 == 0:
        os.makedirs("saved_models", exist_ok=True)
        torch.save(generator.state_dict(), f"saved_models/generator_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"saved_models/discriminator_{epoch}.pth")
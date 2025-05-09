#!/usr/bin/env python3
"""
VAE Image Generator Script

This script loads a pre-trained Variational Autoencoder (VAE) model and uses it to generate
synthetic images by reconstructing images from an input directory.

Usage:
    python vae_image_generator.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
                                  [--model_path MODEL_PATH] [--num_images NUM_IMAGES]
                                  [--batch_size BATCH_SIZE] [--img_size IMG_SIZE]
                                  [--latent_dim LATENT_DIM]

Arguments:
    --input_dir: Directory containing input images for reconstruction
    --output_dir: Directory to save generated images
    --model_path: Path to the saved VAE model (default: vae_model_480_epoch100.pth)
    --num_images: Number of images to generate (default: 10, 0 for all images in directory)
    --batch_size: Batch size for processing (default: 32)
    --img_size: Size to resize images to (default: 480)
    --latent_dim: Dimension of latent space (default: 128)
"""

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm


# Define the VAE model architecture (matching the one from training)
class VAE(nn.Module):
    def __init__(self, latent_dim, img_size=480):
        super(VAE, self).__init__()

        # For 480x480 input, we need appropriate downsampling layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # [B, 32, 240, 240]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # [B, 64, 120, 120]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # [B, 128, 60, 60]
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # [B, 256, 30, 30]
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),  # [B, 512, 15, 15]
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the flattened feature dimensions: 512 * 15 * 15
        self.fc_mu = nn.Linear(512 * 15 * 15, latent_dim)
        self.fc_logvar = nn.Linear(512 * 15 * 15, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 15 * 15)

        # For 480x480 output, we need appropriate upsampling layers
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 15, 15)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # [B, 256, 30, 30]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # [B, 128, 60, 60]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # [B, 64, 120, 120]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # [B, 32, 240, 240]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # [B, 3, 480, 480]
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)
        z = self.reparameterize(mu, logvar)
        x_dec = self.fc_decode(z)
        x_recon = self.decoder(x_dec)
        return x_recon, mu, logvar

    def encode(self, x):
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)
        return mu, logvar

    def decode(self, z):
        x_dec = self.fc_decode(z)
        x_recon = self.decoder(x_dec)
        return x_recon

    def generate(self, x):
        # Simply encode and decode without sampling
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)


# Custom dataset for loading images
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img_name = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


def generate_images(args):
    # Set device
    seed=1583891
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [-1, 1] range
    ])

    # Create dataset and dataloader
    dataset = ImageDataset(args.input_dir, transform=transform)

    if args.num_images == 0:
        # Process all images
        num_images = len(dataset)
    else:
        # Process only the specified number of images
        num_images = min(args.num_images, len(dataset))

    # If we're selecting a subset of images, choose them randomly
    if num_images < len(dataset):
        indices = random.sample(range(len(dataset)), num_images)
        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loading model from {args.model_path}")

    # Initialize and load model
    model = VAE(args.latent_dim, args.img_size).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print(f"Generating {num_images} reconstructed images...")

    # Generate images
    with torch.no_grad():
        for batch_idx, (imgs, img_names) in enumerate(tqdm(dataloader)):
            imgs = imgs.to(device)

            # Generate reconstructions
            recons, _, _ = model(imgs)

            # Save each image in the batch
            for i, (recon, img_name) in enumerate(zip(recons, img_names)):
                # De-normalize from [-1, 1] to [0, 1]
                recon = (recon + 1) / 2.0

                # Add a prefix to differentiate from original images
                output_name = f"vae_{img_name}"
                output_path = os.path.join(args.output_dir, output_name)

                # Save the image
                save_image(recon, output_path)

    print(f"Done! {num_images} images generated and saved to {args.output_dir}")


if __name__ == "__main__":
    from types import SimpleNamespace

    # Shared settings
    common_args = {
        "output_dir": r"E:\Datasets\masati-thesis\synthetic_images\4_vae_generation",
        "model_path": r"E:\PycharmProjects\synthetic-satellite-pipeline\models\vae_model_480_epoch100.pth",
        "batch_size": 32,
        "img_size": 480,
        "latent_dim": 128
    }

    # Dataset-specific settings
    datasets = [
        {"input_dir": r"E:\Datasets\masati-thesis\synthetic_images\0_original_images", "num_images": 750},
        {"input_dir": r"E:\Datasets\masati-thesis\synthetic_images\1_object_insertion", "num_images": 2000},
        {"input_dir": r"E:\Datasets\masati-thesis\synthetic_images\2_augmentation", "num_images": 2500},
        {"input_dir": r"E:\Datasets\masati-thesis\synthetic_images\3_cloud_generation", "num_images": 750},
    ]

    # Run VAE generation for each dataset
    for config in datasets:
        args = SimpleNamespace(**common_args, **config)
        generate_images(args)

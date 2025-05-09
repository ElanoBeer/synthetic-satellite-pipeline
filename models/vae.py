import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time

# --- Settings ---
img_size = 480  # Updated from 224 to 480
batch_size = 32
latent_dim = 128
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transformations ---
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1] range
])


# --- Custom Dataset for Unlabeled Images from Multiple Directories ---
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, roots, transform=None):
        self.files = []
        # Handle both single directory (string) and multiple directories (list)
        if isinstance(roots, str):
            roots = [roots]

        # Collect files from all directories
        for root in roots:
            files_in_dir = [os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('.jpg', '.png'))]
            self.files.extend(files_in_dir)

        self.transform = transform
        print(f"Total images found: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        return img


image_dirs = [
    r"E:/Datasets/masati-thesis/synthetic_images/0_original_images",
    r"E:/Datasets/masati-thesis/synthetic_images/1_object_insertion",
    r"E:/Datasets/masati-thesis/synthetic_images/2_augmentation",
    r"E:/Datasets/masati-thesis/synthetic_images/3_cloud_generation",
]
dataset = ImageDataset(image_dirs, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# --- VAE Architecture ---
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        # For 480x480 input, we need an additional downsampling layer
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

        # For 480x480 output, we need an additional upsampling layer
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


# --- Loss Function ---
def vae_loss(x, x_recon, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


if __name__ == "__main__":
    model = VAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- Training Loop ---
    # For tracking and visualizing loss
    epoch_losses = []
    batch_losses = []
    start_time = time.time()
    log_interval = 10  # Log every 10 batches

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                            desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, imgs in progress_bar:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss = vae_loss(imgs, recon, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() / len(imgs)
            total_loss += loss.item()

            # Track batch loss
            batch_losses.append(batch_loss)

            # Update progress bar with current batch loss
            progress_bar.set_postfix({
                'batch_loss': f'{batch_loss:.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1) / batch_size:.4f}'
            })

            # Print more detailed updates at regular intervals
            if (batch_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"\nBatch {batch_idx + 1}/{len(dataloader)} | "
                      f"Loss: {batch_loss:.4f} | "
                      f"Time: {elapsed:.2f}s")

        # Calculate and store epoch loss
        epoch_loss = total_loss / len(dataset)
        epoch_losses.append(epoch_loss)
        epoch_time = time.time() - epoch_start

        print(f"[Epoch {epoch + 1}/{epochs}] "
              f"Loss: {epoch_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")

        # Optional: Save checkpoint every few epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"vae_model_480_epoch{epoch + 1}.pth")
            print(f"Checkpoint saved at epoch {epoch + 1}")

    # --- Save the Final Model ---
    torch.save(model.state_dict(), "vae_model_480.pth")
    print("Final model saved as 'vae_model_480.pth'")

    # --- Visualize Loss ---
    plt.figure(figsize=(12, 5))

    # Plot epoch losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), epoch_losses, 'b-')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot batch losses
    plt.subplot(1, 2, 2)
    plt.plot(batch_losses, 'r-')
    plt.title('Training Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('vae_training_loss.png', dpi=300)
    plt.show()
    print("Loss visualization saved as 'vae_training_loss.png'")
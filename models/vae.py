import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from PIL import Image
from tqdm import tqdm

# --- Settings ---
img_size = 224
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

# --- Custom Dataset for Unlabeled Images ---
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        return img

# Replace with your image folder path
dataset = ImageDataset("E:/Datasets/masati-thesis/augmented-images", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# --- VAE Architecture ---
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # [B, 32, 112, 112]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # [B, 64, 56, 56]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # [B, 128, 28, 28]
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128 * 28 * 28, latent_dim)
        self.fc_logvar = nn.Linear(128 * 28 * 28, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 28 * 28)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 28, 28)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
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
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss = vae_loss(imgs, recon, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(dataset):.4f}")

    # --- Save the Model ---
    torch.save(model.state_dict(), "vae_model.pth")
    print("Model saved as 'vae_model.pth'")
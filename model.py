"""
model.py — Autoencoder CNN per audio 

PERCHÉ AUTOENCODER E NON VAE
─────────────────────────────
Il VAE aggiunge una componente probabilistica (KL divergence) che su dataset
piccoli tende a collassare: il modello impara a ignorare il latent space e
ricostruisce tutto dalla media. Risultato: latent space piatto, ricostruzioni
sfocate, interpolazioni che non funzionano.

Un autoencoder puro:
  - Converge più velocemente
  - Latent space più "carico" di informazione
  - Interpolazioni più fluide
  - Meno hyperparameter da tuninare

NOISE INJECTION
───────────────
Durante il training, aggiungiamo rumore gaussiano al latent vector:
  z_noisy = z + 0.1 * N(0,1)

Questo forza il decoder a essere robusto a piccole perturbazioni del latent,
rendendo lo spazio più "liscio" e le interpolazioni più fluide.

FLUSSO DATI
───────────
  Spettrogramma (1, 100, 64)
       │
       ▼
  [Encoder CNN]
       │
       ▼
  latent vector (16 dim)
       │ + rumore durante training
       ▼
  [Decoder CNN]
       │
       ▼
  Spettrogramma ricostruito (1, 100, 64)
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Comprime lo spettrogramma in un vettore latente.
    Input:  (batch, 1, 100, 64)
    Output: (batch, latent_dim)

    Dopo 3x Conv2d stride=2:
      100 → 50 → 25 → 13
       64 → 32 → 16 →  8
    → flatten: 64 * 13 * 8 = 6656
    """
    def __init__(self, latent_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # → (16, 50, 32)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # → (32, 25, 16)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → (64, 13, 8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(64 * 13 * 8, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    """
    Ricostruisce lo spettrogramma dal vettore latente.
    Input:  (batch, latent_dim)
    Output: (batch, 1, 100, 64)

    Le ConvTranspose2d con stride=2 producono:
      13 → 26 → 52 → 104   (dim mel)
       8 → 16 → 32 →  64   (dim tempo)
    Il crop finale riporta 104 → 100.
    """
    def __init__(self, latent_dim=16):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 13 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # → (32, 26, 16)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # → (16, 52, 32)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # → (1, 104, 64)
        )

    def forward(self, z):
        x = torch.relu(self.fc(z))
        x = x.view(-1, 64, 13, 8)
        x = self.deconv(x)
        x = x[:, :, :100, :64]  
        return x


class Autoencoder(nn.Module):
    """
    Autoencoder CNN per mel-spettrogrammi.

    noise_std: rumore aggiunto al latent durante training
               rende lo spazio più smooth e le interpolazioni più fluide
    """
    def __init__(self, latent_dim=16, noise_std=0.1):
        super().__init__()
        self.encoder    = Encoder(latent_dim)
        self.decoder    = Decoder(latent_dim)
        self.latent_dim = latent_dim
        self.noise_std  = noise_std

    def forward(self, x):
        z = self.encoder(x)
        if self.training and self.noise_std > 0:
            z = z + self.noise_std * torch.randn_like(z)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def interpolate(self, z1, z2, t):
        """Interpolazione lineare tra due suoni. t=0→z1, t=1→z2"""
        return self.decode((1 - t) * z1 + t * z2)


def ae_loss(x_recon, x_original):
    """MSE pura — niente KL divergence."""
    return nn.functional.mse_loss(x_recon, x_original)


if __name__ == "__main__":
    dummy = torch.randn(4, 1, 100, 64)
    model = Autoencoder(latent_dim=16)
    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")
    model.train()
    recon, z = model(dummy)
    print(f"Input: {dummy.shape} → Latent: {z.shape} → Output: {recon.shape}")
    assert recon.shape == dummy.shape, f"Shape mismatch: {recon.shape} vs {dummy.shape}"
    print(f"Loss: {ae_loss(recon, dummy).item():.4f}")
    print("OK!")
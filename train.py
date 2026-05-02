"""
train.py — Training dell'Autoencoder audio


Uso:
  python train.py
  python train.py --data_dir ./data_processed  --epochs 200
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import umap
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model   import Autoencoder, ae_loss
from dataset import get_dataloader

CONFIG = {
    "data_dir":       "./data_processed",
    "max_samples":    None,
    "batch_size":     32,
    "latent_dim":     16,
    "noise_std":      0.1,
    "epochs":         200,
    "learning_rate":  1e-3,
    "checkpoint_dir": "./checkpoints",
    "save_every":     10,
    "plot_every":     5,
}

# Palette colori per tracce
TRACK_PALETTE = [
    "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#ff6bdb",
    "#ff9f43", "#48dbfb", "#ff9ff3", "#54a0ff", "#5f27cd",
    "#00d2d3", "#ff6348",
]


def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("Apple Silicon (MPS)")
    else:
        d = torch.device("cpu")
        print("CPU")
    return d


def plot_latent_pca_umap(model, dataset, device, epoch, save_dir):
    """
    Due pannelli affiancati: sinistra PCA, destra UMAP.
    Entrambi colorati per traccia sorgente, stile dark.
    """
    from sklearn.decomposition import PCA as _PCA
    model.eval()
    all_z, instruments = [], []

    with torch.no_grad():
        for i in range(len(dataset)):
            s    = dataset[i]
            spec = s['spectrogram'].unsqueeze(0).to(device)
            z    = model.encode(spec).cpu().numpy()[0]
            all_z.append(z)
            instruments.append(s['instrument'])

    all_z = np.array(all_z)

    # PCA: 16D → 2D
    pca   = _PCA(n_components=2)
    z_pca = pca.fit_transform(all_z)
    var   = pca.explained_variance_ratio_

    # UMAP: 16D → 2D
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.2,
        metric="euclidean",
        random_state=42,
        low_memory=True,
    )
    z_umap = reducer.fit_transform(all_z)

    inst_list = sorted(set(instruments))
    inst_to_i = {inst: i for i, inst in enumerate(inst_list)}

    BG       = "#0d0d0d"
    GRID_COL = "#1a1a1a"
    TEXT_COL = "#888888"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG)

    for ax, z_2d, title, xlabel, ylabel in [
        (axes[0], z_pca,
         f"PCA — epoch {epoch}  ({var[0]:.1%} + {var[1]:.1%})",
         f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})"),
        (axes[1], z_umap,
         f"UMAP — epoch {epoch}",
         "UMAP 1", "UMAP 2"),
    ]:
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID_COL, linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

        for inst in inst_list:
            mask  = [m == inst for m in instruments]
            pts   = z_2d[mask]
            color = TRACK_PALETTE[inst_to_i[inst] % len(TRACK_PALETTE)]
            ax.scatter(pts[:, 0], pts[:, 1],
                       color=color, alpha=0.7, s=12,
                       linewidths=0, zorder=2, label=inst)

        ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9, fontfamily="monospace")
        ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9, fontfamily="monospace")
        ax.set_title(title, color="#dddddd", fontsize=10,
                     fontfamily="monospace", loc="left", pad=8)
        ax.tick_params(colors="#444444", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)
        ax.legend(loc="lower right", fontsize=6,
                  framealpha=0.7, labelcolor="#cccccc",
                  facecolor="#111111", edgecolor="#333333",
                  markerscale=1.6)

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout(pad=1.2)
    plt.savefig(f"{save_dir}/latent_epoch_{epoch:03d}.png", dpi=150, facecolor=BG)
    plt.close()
    print(f"  → PCA + UMAP plot salvato")


def train(config=CONFIG):
    device = get_device()
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    plots_dir = os.path.join(config["checkpoint_dir"], "plots")

    loader, dataset = get_dataloader(
        config["data_dir"],
        batch_size=config["batch_size"],
        max_samples=config["max_samples"],
    )

    model = Autoencoder(
        latent_dim=config["latent_dim"],
        noise_std=config["noise_std"]
    ).to(device)

    print(f"Autoencoder: {sum(p.numel() for p in model.parameters()):,} parametri")
    print(f"Latent dim: {config['latent_dim']}  |  Noise std: {config['noise_std']}")

    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = []

    print(f"\n{'Epoch':>6} | {'Loss':>10} | {'LR':>10}")
    print("-" * 32)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0

        for batch in loader:
            specs = batch['spectrogram'].to(device)
            recon, z = model(specs)
            loss = ae_loss(recon, specs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        history.append(epoch_loss)
        scheduler.step(epoch_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6} | {epoch_loss:>10.4f} | {lr:>10.6f}")

        if epoch % config["save_every"] == 0 or epoch == config["epochs"]:
            ckpt = {
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'config':           config,
                'loss':             epoch_loss,
                'dataset_mean':     dataset.mean,
                'dataset_std':      dataset.std,
            }
            path = f"{config['checkpoint_dir']}/ae_epoch_{epoch:03d}.pt"
            torch.save(ckpt, path)
            print(f"  → Checkpoint: {path}")

        if epoch % config["plot_every"] == 0 or epoch == config["epochs"]:
            plot_latent_pca_umap(model, dataset, device, epoch, plots_dir)

    # Salva finale
    final = f"{config['checkpoint_dir']}/ae_final.pt"
    torch.save({
        'epoch':            config["epochs"],
        'model_state_dict': model.state_dict(),
        'config':           config,
        'loss':             history[-1],
        'dataset_mean':     dataset.mean,
        'dataset_std':      dataset.std,
    }, final)
    print(f"\nModello finale: {final}")

    # Curva loss
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.plot(history, color="#ff6b6b", linewidth=1.5, label="MSE Loss")
    ax.set_xlabel("Epoch", color="#888888", fontfamily='monospace')
    ax.set_ylabel("Loss",  color="#888888", fontfamily='monospace')
    ax.set_title("TRAINING LOSS", color="#dddddd",
                 fontfamily='monospace', loc='left')
    ax.tick_params(colors='#444444')
    ax.grid(alpha=0.15, color='#ffffff')
    for sp in ax.spines.values(): sp.set_edgecolor("#1a1a1a")
    ax.legend(fontsize=8, labelcolor='#cccccc',
              facecolor='#111111', edgecolor='#333333')
    plt.tight_layout()
    plt.savefig(f"{config['checkpoint_dir']}/training_loss.png",
                dpi=150, facecolor="#0d0d0d")
    plt.close()

    print("Training completato!")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default=CONFIG["data_dir"])
    parser.add_argument("--max_samples", type=int, default=CONFIG["max_samples"])
    parser.add_argument("--epochs",      type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch_size",  type=int, default=CONFIG["batch_size"])
    parser.add_argument("--latent_dim",  type=int, default=CONFIG["latent_dim"])
    parser.add_argument("--noise_std",   type=float, default=CONFIG["noise_std"])
    args = parser.parse_args()
    CONFIG.update(vars(args))
    train(CONFIG)

"""
export_umap_points.py — Estrae le coordinate UMAP 2D dal modello e le salva in un JSON.

Uso:
  python connessioneMAX/export_umap_points.py --checkpoint ./checkpoints/ae_final.pt \
                                               --data_dir   ./data_processed \
                                               --output     ./umap_points.json

Il file output contiene:
  {
    "points": [[x, y], ...],          # coordinate UMAP 2D
    "instruments": ["bass", ...],     # label strumento per ogni punto
    "filenames": ["bass_001.wav", ...]
  }
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import umap

# The scripts live one level below the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import NSynthDataset
from model   import Autoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "ae_final.pt"))
    parser.add_argument("--data_dir",   default=str(PROJECT_ROOT / "data_processed"))
    parser.add_argument("--output",     default=str(PROJECT_ROOT / "umap_points.json"))
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Carica modello ────────────────────────────────────────────────────────
    ckpt    = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config  = ckpt['config']
    ds_mean = ckpt['dataset_mean']
    ds_std  = ckpt['dataset_std']

    model = Autoencoder(latent_dim=config['latent_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Modello caricato — epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")

    # ── Encode dataset ────────────────────────────────────────────────────────
    dataset = NSynthDataset(args.data_dir, cache=False, mean=ds_mean, std=ds_std)
    print(f"Dataset: {len(dataset)} campioni")

    zs, instruments, filenames = [], [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            s    = dataset[i]
            spec = s['spectrogram'].unsqueeze(0).to(device)
            z    = model.encode(spec).cpu().numpy()[0]
            zs.append(z)
            instruments.append(s['instrument'])
            filenames.append(s['filename'])
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(dataset)}")

    Z = np.array(zs)
    print(f"Latent shape: {Z.shape}")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print("Calcolo UMAP...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=20, min_dist=0.5,
        metric='euclidean', random_state=42, low_memory=True,
    )
    Z2D = reducer.fit_transform(Z)
    print("UMAP completata")

    # ── Salva ─────────────────────────────────────────────────────────────────
    out = {
        "points":      Z2D.tolist(),
        "instruments": instruments,
        "filenames":   filenames,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f)

    print(f"\nSalvato in: {args.output}")
    print(f"  {len(instruments)} punti")
    print(f"  strumenti: {sorted(set(instruments))}")


if __name__ == "__main__":
    main()

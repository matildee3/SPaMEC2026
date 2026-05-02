"""
prerender.py — Pre-renderizza una griglia di suoni dal latent space

Divide lo spazio UMAP in una griglia N×N, genera un suono per ogni cella
via decoder + Griffin-Lim, e salva tutto in una cartella.

Uso:
  python prerender.py --checkpoint ./checkpoints/ae_final.pt \
                      --data_dir   ./data_processed \
                      --grid_size  20 \
                      --n_iter     32

Parametri chiave:
  --grid_size   dimensione della griglia (20 → 400 suoni, 30 → 900 suoni)
  --n_iter      iterazioni Griffin-Lim (128=qualità, 32=veloce, 8=molto veloce)
  --output_dir  dove salvare i file (default: ./grid_audio)

Output:
  grid_audio/
  ├── grid_manifest.json   — mappa coordinate → file audio
  ├── 000_000.wav
  ├── 000_001.wav
  └── ...
"""

import argparse
import json
import os
import numpy as np
import torch
import librosa
import soundfile as sf
import umap
from pathlib import Path

from model   import Autoencoder
from dataset import NSynthDataset, SAMPLE_RATE, N_FFT, HOP_LENGTH


def setup(checkpoint_path, data_dir):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt    = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config  = ckpt['config']
    ds_mean = ckpt['dataset_mean']
    ds_std  = ckpt['dataset_std']

    model = Autoencoder(latent_dim=config['latent_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Modello caricato — epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")

    dataset = NSynthDataset(data_dir, cache=True, mean=ds_mean, std=ds_std)

    print("Calcolo coordinate latenti...")
    all_z, all_meta = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            s    = dataset[i]
            spec = s['spectrogram'].unsqueeze(0).to(device)
            z    = model.encode(spec).cpu().numpy()[0]
            all_z.append(z)
            all_meta.append({'instrument': s['instrument'], 'filename': s['filename']})

    all_z = np.array(all_z)

    print("Calcolo UMAP...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.2,
        metric='euclidean', random_state=42, low_memory=True,
    )
    z_umap = reducer.fit_transform(all_z)
    print(f"UMAP completata — X: [{z_umap[:,0].min():.2f}, {z_umap[:,0].max():.2f}]  "
          f"Y: [{z_umap[:,1].min():.2f}, {z_umap[:,1].max():.2f}]")

    return model, device, all_z, z_umap, all_meta, ds_mean, ds_std


def decode_to_audio(model, device, z_16d, ds_mean, ds_std, n_iter):
    """Latent 16D → waveform audio via decoder + Griffin-Lim."""
    z_t = torch.tensor(z_16d[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        spec_t = model.decode(z_t)

    spec_np  = spec_t[0, 0].cpu().numpy()
    spec_log = spec_np * ds_std + ds_mean
    mel      = np.exp(spec_log) - 1e-5
    mel      = np.maximum(mel, 0)
    p99      = np.percentile(mel, 99)
    if p99 > 0:
        mel = np.clip(mel, 0, p99 * 2)

    waveform = librosa.feature.inverse.mel_to_audio(
        mel, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_iter=n_iter, power=1.0
    )
    mx = np.abs(waveform).max()
    if mx > 0:
        waveform = waveform / mx

    return waveform.astype(np.float32)


def prerender_grid(model, device, all_z, z_umap, ds_mean, ds_std,
                   grid_size, n_iter, output_dir):
    """
    Genera un suono per ogni cella della griglia N×N nello spazio UMAP.
    Per ogni cella usa l'interpolazione pesata dei k vicini più prossimi.
    """
    os.makedirs(output_dir, exist_ok=True)

    x_min, x_max = z_umap[:, 0].min(), z_umap[:, 0].max()
    y_min, y_max = z_umap[:, 1].min(), z_umap[:, 1].max()

    # Aggiungi un margine del 5%
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    x_min -= margin_x; x_max += margin_x
    y_min -= margin_y; y_max += margin_y

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)

    manifest = {
        "grid_size":  grid_size,
        "n_iter":     n_iter,
        "x_min":      float(x_min),
        "x_max":      float(x_max),
        "y_min":      float(y_min),
        "y_max":      float(y_max),
        "sample_rate": SAMPLE_RATE,
        "cells":      {},
    }

    total = grid_size * grid_size
    print(f"\nGenerazione griglia {grid_size}×{grid_size} = {total} suoni")
    print(f"Griffin-Lim iterazioni: {n_iter}")
    print(f"Output: {output_dir}/\n")

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            idx_flat = i * grid_size + j
            if (idx_flat + 1) % 50 == 0 or idx_flat == 0:
                print(f"  {idx_flat+1}/{total}  ({(idx_flat+1)/total*100:.0f}%)")

            # Interpolazione pesata dai 4 vicini più prossimi nel 2D
            distances = np.linalg.norm(z_umap - np.array([x, y]), axis=1)
            k         = 4
            k_idx     = np.argsort(distances)[:k]
            k_dists   = distances[k_idx]
            weights   = 1.0 / (k_dists + 1e-6)
            weights  /= weights.sum()
            z_16d     = (all_z[k_idx] * weights[:, None]).sum(axis=0)

            waveform = decode_to_audio(
                model, device, z_16d, ds_mean, ds_std, n_iter
            )

            fname = f"{i:03d}_{j:03d}.wav"
            sf.write(os.path.join(output_dir, fname), waveform, SAMPLE_RATE)

            manifest["cells"][f"{i},{j}"] = {
                "file":  fname,
                "x":     float(x),
                "y":     float(y),
                "gi":    i,
                "gj":    j,
            }

    manifest_path = os.path.join(output_dir, "grid_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {total} file salvati in '{output_dir}/'")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints/ae_final.pt")
    parser.add_argument("--data_dir",   default="./data_processed")
    parser.add_argument("--grid_size",  type=int, default=20)
    parser.add_argument("--n_iter",     type=int, default=32,
                        help="Iterazioni Griffin-Lim: 128=qualità, 32=veloce, 8=molto veloce")
    parser.add_argument("--output_dir", default="./grid_audio")
    args = parser.parse_args()

    model, device, all_z, z_umap, all_meta, ds_mean, ds_std = setup(
        args.checkpoint, args.data_dir
    )
    prerender_grid(
        model, device, all_z, z_umap, ds_mean, ds_std,
        grid_size=args.grid_size,
        n_iter=args.n_iter,
        output_dir=args.output_dir,
    )

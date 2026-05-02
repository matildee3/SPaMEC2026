"""

Avvio:
  python demo_umap.py --checkpoint ./checkpoints/ae_final.pt \
                        --data_dir   ./data_processed
"""

import argparse
import numpy as np
import torch
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA
import umap
import librosa

from model   import Autoencoder
from dataset import NSynthDataset, spectrogram_to_audio, SAMPLE_RATE, N_FFT, HOP_LENGTH

MODEL     = None
DATASET   = None
DEVICE    = None
ALL_Z     = None
ALL_META  = None

DS_MEAN   = None
DS_STD    = None

TRACK_PALETTE = [
    "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#ff6bdb",
    "#ff9f43", "#48dbfb", "#ff9ff3", "#54a0ff", "#5f27cd",
    "#00d2d3", "#ff6348",
]


def setup(checkpoint_path, data_dir, max_samples):
    global MODEL, DATASET, DEVICE, ALL_Z, ALL_META, DS_MEAN, DS_STD, UMAP_MODEL, Z2D_CACHE

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    ckpt    = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config  = ckpt['config']
    DS_MEAN = ckpt['dataset_mean']
    DS_STD  = ckpt['dataset_std']

    MODEL = Autoencoder(latent_dim=config['latent_dim']).to(DEVICE)
    MODEL.load_state_dict(ckpt['model_state_dict'])
    MODEL.eval()
    print(f"Modello caricato — epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")

    DATASET = NSynthDataset(data_dir, max_samples=max_samples,
                            cache=True, mean=DS_MEAN, std=DS_STD)

    print("Calcolo coordinate latenti...")
    zs, metas = [], []
    with torch.no_grad():
        for i in range(len(DATASET)):
            s    = DATASET[i]
            spec = s['spectrogram'].unsqueeze(0).to(DEVICE)
            z    = MODEL.encode(spec).cpu().numpy()[0]
            zs.append(z)
            metas.append({
                'instrument': s['instrument'],
                'pitch':      s['pitch'],
                'filename':   s['filename'],
                'idx':        i,
            })
    ALL_Z    = np.array(zs)
    ALL_META = metas

    print("Calcolo UMAP...")
    UMAP_MODEL = umap.UMAP(
        n_components=2,
        n_neighbors=20, #15
        min_dist=0.5, #0.1
        metric='euclidean',
        random_state=42,
    )
    UMAP_MODEL.fit(ALL_Z)
    Z2D_CACHE = UMAP_MODEL.transform(ALL_Z)
    print(f"UMAP completata — range X: [{Z2D_CACHE[:,0].min():.1f}, {Z2D_CACHE[:,0].max():.1f}]")
    print(f"Pronti: {len(ALL_Z)} suoni")


def get_z2d():
    return Z2D_CACHE


def load_wav_direct(idx):
    fname    = ALL_META[idx]['filename']
    wav_path = str(DATASET.audio_dir / f"{fname}.wav")
    wave, _  = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True, duration=1.0)
    return (SAMPLE_RATE, (wave * 32767).astype(np.int16))


def make_plot(highlight_real=None, highlight_gen_2d=None):
    z2d         = get_z2d()
    instruments = sorted(set(m['instrument'] for m in ALL_META))
    inst_to_i   = {inst: i for i, inst in enumerate(instruments)}

    BG       = "#0d0d0d"
    GRID_COL = "#1a1a1a"
    TICK_COL = "#444444"
    TEXT_COL = "#888888"

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.grid(True, color=GRID_COL, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    for inst in instruments:
        mask   = [m['instrument'] == inst for m in ALL_META]
        pts    = z2d[mask]
        color  = TRACK_PALETTE[inst_to_i[inst] % len(TRACK_PALETTE)]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=color, alpha=0.65, s=14,
                   linewidths=0, zorder=2, label=inst)

    if highlight_real is not None:
        p     = z2d[highlight_real]
        inst  = ALL_META[highlight_real]['instrument']
        color = TRACK_PALETTE[inst_to_i[inst] % len(TRACK_PALETTE)]
        ax.scatter(p[0], p[1], color=color, s=220, zorder=6,
                   marker='o', linewidths=1.5, edgecolors='white',
                   path_effects=[pe.withSimplePatchShadow(offset=(0, 0),
                                 shadow_rgbFace='white', alpha=0.25)])
        ax.scatter(p[0], p[1], color='white', s=40, zorder=7, marker='o')

    if highlight_gen_2d is not None:
        gx, gy = highlight_gen_2d
        ax.scatter(gx, gy, color='#ffffff', s=160, zorder=8,
                   marker='+', linewidths=2)
        circle = plt.Circle((gx, gy), radius=((z2d[:, 0].max() - z2d[:, 0].min()) * 0.03),
                             fill=False, color='#ffffff', linewidth=0.8,
                             alpha=0.4, zorder=7)
        ax.add_patch(circle)

    ax.set_xlabel("UMAP 1", color=TEXT_COL, fontsize=9, fontfamily='monospace')
    ax.set_ylabel("UMAP 2", color=TEXT_COL, fontsize=9, fontfamily='monospace')
    ax.set_title("LATENT SPACE  (UMAP)", color="#dddddd", fontsize=11,
                 fontfamily='monospace', loc='left', pad=10)

    ax.tick_params(colors=TICK_COL, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    legend = ax.legend(
        loc='lower right', fontsize=7,
        framealpha=0.7, labelcolor='#cccccc',
        facecolor='#111111', edgecolor='#333333',
        markerscale=1.4,
    )

    plt.tight_layout(pad=1.2)
    plt.close(fig)
    return fig


def click_nearest(x_click, y_click):
    z2d       = get_z2d()
    distances = np.linalg.norm(z2d - np.array([x_click, y_click]), axis=1)
    idx       = int(np.argmin(distances))
    meta      = ALL_META[idx]
    audio     = load_wav_direct(idx)
    fig       = make_plot(highlight_real=idx)
    info      = (f"traccia: {meta['instrument']}  ·  "
                 f"file: {meta['filename']}  ·  "
                 f"distanza: {distances[idx]:.3f}")
    return audio, fig, info


def decode_from_coords(x_val, y_val):
    z2d = get_z2d()
    distances_2d = np.linalg.norm(z2d - np.array([x_val, y_val]), axis=1)
    k = 4
    k_idx     = np.argsort(distances_2d)[:k]
    k_dists   = distances_2d[k_idx]
    weights   = 1.0 / (k_dists + 1e-6)
    weights  /= weights.sum()
    z_16d     = (ALL_Z[k_idx] * weights[:, None]).sum(axis=0, keepdims=True)
    z_t       = torch.tensor(z_16d, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        spec_t = MODEL.decode(z_t)

    spec_np  = spec_t[0, 0].cpu().numpy()
    spec_log = spec_np * DS_STD + DS_MEAN
    mel      = np.exp(spec_log) - 1e-5
    mel      = np.maximum(mel, 0)
    p99      = np.percentile(mel, 99)
    if p99 > 0:
        mel = np.clip(mel, 0, p99 * 2)

    waveform = librosa.feature.inverse.mel_to_audio(
        mel, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_iter=128, power=1.0
    )
    mx = np.abs(waveform).max()
    if mx > 0:
        waveform = waveform / mx

    audio = (SAMPLE_RATE, (waveform * 32767).astype(np.int16))

    distances = np.linalg.norm(z2d - np.array([x_val, y_val]), axis=1)
    idx       = int(np.argmin(distances))
    meta      = ALL_META[idx]
    fig       = make_plot(highlight_gen_2d=(x_val, y_val), highlight_real=idx)
    info      = (f"generato dal decoder  ·  "
                 f"reale più vicino: {meta['instrument']}  ·  "
                 f"distanza: {distances[idx]:.2f}")
    return audio, fig, info


def nearest_from_coords(x_val, y_val):
    """2D → suono reale più vicino."""
    z2d       = get_z2d()
    distances = np.linalg.norm(z2d - np.array([x_val, y_val]), axis=1)
    idx       = int(np.argmin(distances))
    meta      = ALL_META[idx]
    audio     = load_wav_direct(idx)
    fig       = make_plot(highlight_gen_2d=(x_val, y_val), highlight_real=idx)
    info      = (f"reale più vicino: {meta['instrument']}  ·  "
                 f"file: {meta['filename']}  ·  "
                 f"distanza: {distances[idx]:.3f}")
    return audio, fig, info


def build_ui():
    z2d     = get_z2d()
    z_min_x = float(z2d[:, 0].min()) - 2
    z_max_x = float(z2d[:, 0].max()) + 2
    z_min_y = float(z2d[:, 1].min()) - 2
    z_max_y = float(z2d[:, 1].max()) + 2
    z_mid_x = (z_min_x + z_max_x) / 2
    z_mid_y = (z_min_y + z_max_y) / 2

    css = """
    body, .gradio-container {
        background-color: #0d0d0d !important;
        color: #cccccc !important;
        font-family: 'Courier New', monospace !important;
    }
    .gr-button-primary {
        background: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        letter-spacing: 0.05em !important;
    }
    .gr-button-secondary {
        background: transparent !important;
        color: #aaaaaa !important;
        border: 1px solid #333333 !important;
        font-family: 'Courier New', monospace !important;
        letter-spacing: 0.05em !important;
    }
    label, .gr-form label {
        color: #666666 !important;
        font-size: 10px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-family: 'Courier New', monospace !important;
    }
    """

    with gr.Blocks(title="Autoencoder Audio") as ui:
        gr.Markdown("## Latent Space Explorer")
        gr.Markdown(
            "**Inserisci coordinate** per trovare il suono più vicino. "
            "**Muovi gli slider** per navigare lo spazio."
        )

        with gr.Row():

            with gr.Column(scale=3):
                plot = gr.Plot(value=make_plot(), label="")

            with gr.Column(scale=2):

                gr.Markdown("### -> Cerca per coordinate")
                with gr.Row():
                    click_x = gr.Number(label="X", value=round(z_mid_x, 2), precision=3)
                    click_y = gr.Number(label="Y", value=round(z_mid_y, 2), precision=3)
                click_btn   = gr.Button("TROVA SUONO REALE", variant="primary")
                click_audio = gr.Audio(label="suono", type="numpy")

                gr.Markdown("---")

                gr.Markdown("### -> Esplora con slider")
                slider_x = gr.Slider(z_min_x, z_max_x, value=z_mid_x,
                                     step=0.5, label="X")
                slider_y = gr.Slider(z_min_y, z_max_y, value=z_mid_y,
                                     step=0.5, label="Y")

                with gr.Row():
                    real_btn = gr.Button("SUONO REALE", variant="primary")
                    gen_btn  = gr.Button("GENERA (decoder)", variant="secondary")

                with gr.Row():
                    real_audio = gr.Audio(label="reale", type="numpy")
                    gen_audio  = gr.Audio(label="generato", type="numpy")

                gr.Markdown("---")
                info_box = gr.Textbox(label="INFO", lines=2,
                                      interactive=False)

        click_btn.click(
            fn=click_nearest,
            inputs=[click_x, click_y],
            outputs=[click_audio, plot, info_box],
        )

        real_btn.click(
            fn=nearest_from_coords,
            inputs=[slider_x, slider_y],
            outputs=[real_audio, plot, info_box],
        )

        gen_btn.click(
            fn=decode_from_coords,
            inputs=[slider_x, slider_y],
            outputs=[gen_audio, plot, info_box],
        )

        slider_x.release(
            fn=lambda x, y: make_plot(highlight_gen_2d=(x, y)),
            inputs=[slider_x, slider_y],
            outputs=[plot],
        )
        slider_y.release(
            fn=lambda x, y: make_plot(highlight_gen_2d=(x, y)),
            inputs=[slider_x, slider_y],
            outputs=[plot],
        )

    return ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="./checkpoints/ae_final.pt")
    parser.add_argument("--data_dir",    default="./data_processed")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--port",        type=int, default=7860)
    parser.add_argument("--share",       action="store_true")
    args = parser.parse_args()

    setup(args.checkpoint, args.data_dir, args.max_samples)
    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share, inbrowser=True)
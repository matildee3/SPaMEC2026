"""

Avvio:
 python demo_pca.py --data_dir ./data_processed --max_samples 4000
                    
"""

import argparse
import numpy as np
import torch
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import librosa

from model   import Autoencoder
from dataset import NSynthDataset, spectrogram_to_audio, SAMPLE_RATE

MODEL    = None
DATASET  = None
DEVICE   = None
ALL_Z    = None
ALL_META = None
PCA_MODEL = None  
DS_MEAN  = None
DS_STD   = None


def setup(checkpoint_path, data_dir, max_samples):
    global MODEL, DATASET, DEVICE, ALL_Z, ALL_META, PCA_MODEL, DS_MEAN, DS_STD

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    ckpt     = torch.load(checkpoint_path, map_location=DEVICE)
    config   = ckpt['config']
    DS_MEAN  = ckpt['dataset_mean']
    DS_STD   = ckpt['dataset_std']

    MODEL = Autoencoder(latent_dim=config['latent_dim']).to(DEVICE)
    MODEL.load_state_dict(ckpt['model_state_dict'])
    MODEL.eval()
    print(f"Modello caricato (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    print(f"Latent dim: {config['latent_dim']}, mean: {DS_MEAN:.3f}, std: {DS_STD:.3f}")

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
            metas.append({'instrument': s['instrument'],
                          'pitch': s['pitch'],
                          'filename': s['filename'],
                          'idx': i})
    ALL_Z    = np.array(zs)
    ALL_META = metas

    # PCA: proietta latent_dim → 2D per la visualizzazione
    PCA_MODEL = PCA(n_components=2)
    PCA_MODEL.fit(ALL_Z)
    print(f"PCA varianza spiegata: {PCA_MODEL.explained_variance_ratio_.sum():.1%}")
    print(f"Pronti: {len(ALL_Z)} suoni")


def get_z2d():
    """Restituisce le coordinate 2D (via PCA) di tutti i suoni."""
    return PCA_MODEL.transform(ALL_Z)


def load_wav_direct(idx):
    """Carica il .wav originale direttamente (niente ricostruzione)."""
    fname    = ALL_META[idx]['filename']
    wav_path = str(DATASET.audio_dir / f"{fname}.wav")
    wave, _  = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True, duration=1.0)
    return (SAMPLE_RATE, (wave * 32767).astype(np.int16))


def make_plot(highlight_real=None, highlight_gen_2d=None):
    z2d        = get_z2d()
    instruments = sorted(set(m['instrument'] for m in ALL_META))
    cmap       = matplotlib.colormaps.get_cmap('tab20').resampled(len(instruments))
    inst_to_i  = {inst: i for i, inst in enumerate(instruments)}
    colors     = [cmap(inst_to_i[m['instrument']]) for m in ALL_META]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    ax.scatter(z2d[:,0], z2d[:,1], c=colors, alpha=0.75, s=18,
               zorder=2, linewidths=0)

    if highlight_real is not None:
        p = z2d[highlight_real]
        ax.scatter(p[0], p[1], c='#ff4444', s=180, zorder=5, marker='*',
                   label=f"▶ {ALL_META[highlight_real]['instrument']}")
        ax.legend(loc='upper right', fontsize=8, framealpha=0.4,
                  labelcolor='white', facecolor='#1a1a2e')

    if highlight_gen_2d is not None:
        ax.scatter(highlight_gen_2d[0], highlight_gen_2d[1],
                   c='#ffff00', s=120, zorder=5, marker='X',
                   edgecolors='black', linewidths=0.5, label='generato')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.4,
                  labelcolor='white', facecolor='#1a1a2e')

    # Legenda strumenti
    handles = [plt.scatter([],[],color=cmap(inst_to_i[i]/len(instruments)),
               label=i, s=30) for i in instruments[:12]]
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              framealpha=0.4, labelcolor='white',
              facecolor='#1a1a2e', edgecolor='#444')

    var = PCA_MODEL.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.0%})", color='#aaaaaa')
    ax.set_ylabel(f"PC2 ({var[1]:.0%})", color='#aaaaaa')
    ax.set_title("Latent Space (PCA 2D)", color='white', pad=10)
    ax.tick_params(colors='#888888')
    for sp in ax.spines.values(): sp.set_edgecolor('#333355')
    ax.grid(alpha=0.15, color='#ffffff')

    plt.tight_layout()
    fig_out = fig
    plt.close(fig)
    return fig_out


def click_nearest(x_click, y_click):
    z2d       = get_z2d()
    query     = np.array([x_click, y_click])
    distances = np.linalg.norm(z2d - query, axis=1)
    idx       = int(np.argmin(distances))
    meta      = ALL_META[idx]

    audio = load_wav_direct(idx)   # .wav originale, niente Griffin-Lim
    fig   = make_plot(highlight_real=idx)
    info  = (f"Strumento: {meta['instrument']}  |  Pitch: {meta['pitch']}  |  "
             f"Distanza: {distances[idx]:.3f}")
    return audio, fig, info


def decode_from_sliders(x_val, y_val):
    """
    Converte il punto 2D (PCA) → latent 16D → decoder → Griffin-Lim.
    Genera un suono NUOVO, non dal dataset.
    """
    z2d   = get_z2d()
    query = np.array([[x_val, y_val]])

    # Inverti la PCA: 2D → 16D
    z_16d = PCA_MODEL.inverse_transform(query)
    z_t   = torch.tensor(z_16d, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        spec_t = MODEL.decode(z_t)

    spec_np  = spec_t[0, 0].cpu().numpy()

    # Denormalizza: spec era in scala (x - mean) / std
    spec_log = spec_np * DS_STD + DS_MEAN

    # Inverti log: exp(x) - 1e-5
    mel = np.exp(spec_log) - 1e-5
    mel = np.maximum(mel, 0)
    p99 = np.percentile(mel, 99)
    if p99 > 0:
        mel = np.clip(mel, 0, p99 * 2)

    import librosa as _librosa
    from dataset import N_FFT, HOP_LENGTH
    waveform = _librosa.feature.inverse.mel_to_audio(
        mel, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_iter=128, power=1.0
    )
    mx = np.abs(waveform).max()
    if mx > 0:
        waveform = waveform / mx

    audio = (SAMPLE_RATE, (waveform * 32767).astype(np.int16))

    # Suono reale più vicino come riferimento
    distances = np.linalg.norm(z2d - np.array([x_val, y_val]), axis=1)
    idx       = int(np.argmin(distances))
    meta      = ALL_META[idx]
    fig       = make_plot(highlight_gen_2d=(x_val, y_val), highlight_real=idx)
    info      = f"Generato dal decoder  |  Reale più vicino: {meta['instrument']} (dist={distances[idx]:.2f})"
    return audio, fig, info


def generate_from_sliders(x_val, y_val):
    """Trova il suono reale più vicino e lo riproduce dal .wav."""
    z2d       = get_z2d()
    query     = np.array([x_val, y_val])
    distances = np.linalg.norm(z2d - query, axis=1)
    idx       = int(np.argmin(distances))
    meta      = ALL_META[idx]

    audio = load_wav_direct(idx)
    fig   = make_plot(highlight_gen_2d=(x_val, y_val), highlight_real=idx)
    info  = (f"Suono reale più vicino: {meta['instrument']}  |  "
             f"Pitch: {meta['pitch']}  |  Distanza: {distances[idx]:.3f}")
    return audio, fig, info


def build_ui():
    z2d     = get_z2d()
    z_min_x = round(float(z2d[:,0].min()) - 0.5, 2)
    z_max_x = round(float(z2d[:,0].max()) + 0.5, 2)
    z_min_y = round(float(z2d[:,1].min()) - 0.5, 2)
    z_max_y = round(float(z2d[:,1].max()) + 0.5, 2)
    z_mid_x = (z_min_x + z_max_x) / 2
    z_mid_y = (z_min_y + z_max_y) / 2

    with gr.Blocks(title="Autoencoder Audio") as ui:
        gr.Markdown("## Latent Space Explorer")
        gr.Markdown(
            "**Inserisci coordinate** per trovare il suono più vicino. "
            "**Muovi gli slider** per navigare lo spazio."
        )

        with gr.Row():
            with gr.Column(scale=3):
                plot = gr.Plot(value=make_plot(), label="Latent Space (PCA 2D)")

            with gr.Column(scale=2):
                gr.Markdown("### -> Cerca per coordinate")
                with gr.Row():
                    click_x = gr.Number(label="X", value=round(z_mid_x,2), precision=3)
                    click_y = gr.Number(label="Y", value=round(z_mid_y,2), precision=3)
                click_btn   = gr.Button("▶ Trova suono", variant="primary")
                click_audio = gr.Audio(label="Suono", type="numpy")

                gr.Markdown("---")
                gr.Markdown("### -> Esplora con slider")
                slider_x = gr.Slider(z_min_x, z_max_x, value=z_mid_x,
                                     step=0.05, label="X")
                slider_y = gr.Slider(z_min_y, z_max_y, value=z_mid_y,
                                     step=0.05, label="Y")
                with gr.Row():
                    real_btn = gr.Button("SUONO REALE", variant="primary")
                    gen_btn  = gr.Button("GENERA (decoder)", variant="secondary")
                with gr.Row():
                    gen_audio  = gr.Audio(label="Suono reale", type="numpy")
                    dec_audio  = gr.Audio(label="Suono generato", type="numpy")
                    

                gr.Markdown("---")
                info_box = gr.Textbox(label="Info", lines=2, interactive=False)

        click_btn.click(fn=click_nearest,
                        inputs=[click_x, click_y],
                        outputs=[click_audio, plot, info_box])

        gen_btn.click(fn=generate_from_sliders,
                      inputs=[slider_x, slider_y],
                      outputs=[gen_audio, plot, info_box])

        real_btn.click(fn=generate_from_sliders,
                       inputs=[slider_x, slider_y],
                       outputs=[gen_audio, plot, info_box])

        gen_btn.click(fn=decode_from_sliders,
                      inputs=[slider_x, slider_y],
                      outputs=[dec_audio, plot, info_box])

        slider_x.release(fn=lambda x,y: make_plot(highlight_gen_2d=(x,y)),
                         inputs=[slider_x, slider_y], outputs=[plot])
        slider_y.release(fn=lambda x,y: make_plot(highlight_gen_2d=(x,y)),
                         inputs=[slider_x, slider_y], outputs=[plot])

    return ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="./checkpoints/ae_final.pt")
    parser.add_argument("--data_dir",    default="./data/nsynth-valid")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--port",        type=int, default=7860)
    parser.add_argument("--share",       action="store_true")
    args = parser.parse_args()

    setup(args.checkpoint, args.data_dir, args.max_samples)
    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share, inbrowser=True)

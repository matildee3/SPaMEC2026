"""
build_dataset.py — Costruisce il dataset da tracce audio personali

COSA FA
───────
Legge tutte le tracce .wav/.aiff da una cartella (default: ./data),
applica onset detection su ciascuna, estrae un segmento di 1 secondo per
ogni onset trovato, scarta i segmenti troppo silenziosi (sotto soglia RMS),
e salva il risultato in una struttura compatibile con dataset.py e train.py.

OUTPUT
──────
  data_processed/
  ├── audio/
  │   ├── traccia1_onset_0000.wav
  │   ├── traccia1_onset_0001.wav
  │   ├── traccia2_onset_0000.wav
  │   └── ...
  └── examples.json
      {
        "traccia1_onset_0000": {
          "source_track": "traccia1",
          "onset_time":    3.41,
          "segment_index": 0,
          "rms":           0.143,
          "instrument_family_str": "traccia1",   ← alias per compatibilità
          "pitch":         0
        },
        ...
      }

PERCHÉ onset detection e non sliding window
────────────────────────────────────────────
Come descritto nel paper, estrarre segmenti dagli onset significa catturare
i momenti di attacco sonoro — la parte più caratteristica e informativa di
ogni evento. I segmenti hanno così più varianza timbrica, il che rende il
latent space più ricco e la navigazione più interessante.

PERCHÉ scartare i silenziosi
─────────────────────────────
Un segmento sotto soglia RMS non porta informazione musicale utile.
Includerlo crea un cluster di "nulla" nel latent space che distorce la
geometria e rende la visualizzazione meno leggibile.

PARAMETRI CHIAVE
────────────────
  --rms_threshold   0.01   Soglia RMS sotto cui il segmento viene scartato.
                           Abbassa se scarti troppo materiale ambient/soft.
                           Alza se passano troppi segmenti quasi-silenziosi.

  --onset_delta     0.07   Sensibilità onset detection (librosa).
                           Abbassa per catturare più onset (anche morbidi).
                           Alza per restare solo sui transient più forti.

  --min_gap         0.5    Distanza minima in secondi tra onset consecutivi.
                           Evita di estrarre segmenti quasi identici da
                           onset ravvicinati (es. burst di batteria).

USO
───
  python build_dataset.py
  python build_dataset.py --input_dir ./data --output_dir ./data_processed
  python build_dataset.py --rms_threshold 0.005 --onset_delta 0.05
"""

import os
import json
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# ── Parametri audio (devono combaciare con dataset.py) ──────────────────────
SAMPLE_RATE = 44100
DURATION    = 1.0          # secondi per segmento
N_SAMPLES   = int(SAMPLE_RATE * DURATION)

# ── Parametri di default ─────────────────────────────────────────────────────
DEFAULT_INPUT_DIR    = "./data"
DEFAULT_OUTPUT_DIR   = "./data_processed"
DEFAULT_RMS_THRESH   = 0.01    # sotto questa soglia → scarta
DEFAULT_ONSET_DELTA  = 0.07    # sensibilità onset detection
DEFAULT_MIN_GAP      = 0.5     # secondi minimi tra onset


def load_audio(path: Path) -> np.ndarray:
    """
    Carica un file audio (wav o aiff), lo porta a mono e lo ricampiona
    a SAMPLE_RATE (16000 Hz). Restituisce array float32.
    """
    waveform, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return waveform


def detect_onsets(waveform: np.ndarray, delta: float, min_gap: float) -> list[float]:
    """
    Rileva gli onset (attacchi sonori) in un waveform.

    Usa librosa.onset.onset_strength per costruire la curva di energia
    nel tempo, poi librosa.onset.onset_detect per individuare i picchi.

    delta:   più alto → solo onset forti; più basso → cattura anche onset morbidi
    min_gap: filtra onset troppo ravvicinati (in secondi)
    """
    # Curva di onset strength
    hop_length = 512
    onset_env  = librosa.onset.onset_strength(
        y=waveform, sr=SAMPLE_RATE, hop_length=hop_length
    )

    # Rilevamento onset (in frame)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=SAMPLE_RATE,
        hop_length=hop_length,
        delta=delta,
        backtrack=True   # porta l'onset all'inizio del transient, non al picco
    )

    # Converti frame → secondi
    onset_times = librosa.frames_to_time(
        onset_frames, sr=SAMPLE_RATE, hop_length=hop_length
    )

    # Filtra onset troppo ravvicinati (min_gap)
    filtered = []
    last_t   = -999.0
    for t in onset_times:
        if t - last_t >= min_gap:
            filtered.append(float(t))
            last_t = t

    return filtered


def extract_segment(waveform: np.ndarray, onset_time: float) -> np.ndarray:
    """
    Estrae N_SAMPLES campioni a partire da onset_time.
    Se il segmento va oltre la fine del file, fa padding con zeri.
    """
    start = int(onset_time * SAMPLE_RATE)
    end   = start + N_SAMPLES

    if start >= len(waveform):
        return np.zeros(N_SAMPLES, dtype=np.float32)

    segment = waveform[start:end]

    if len(segment) < N_SAMPLES:
        segment = np.pad(segment, (0, N_SAMPLES - len(segment)))

    return segment.astype(np.float32)


def compute_rms(segment: np.ndarray) -> float:
    """RMS (Root Mean Square) — misura di energia/ampiezza del segmento."""
    return float(np.sqrt(np.mean(segment ** 2)))


def build_dataset(
    input_dir:    str,
    output_dir:   str,
    rms_threshold: float,
    onset_delta:  float,
    min_gap:      float,
):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    audio_out   = output_path / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    # Estensioni supportate
    extensions = {".wav", ".aiff", ".aif"}

    # Trova tutti i file audio
    track_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in extensions
    ])

    if not track_files:
        print(f"Nessun file audio trovato in {input_dir}")
        print(f"Estensioni cercate: {extensions}")
        return

    print(f"Trovate {len(track_files)} tracce in '{input_dir}'")
    print(f"Parametri: RMS≥{rms_threshold} | delta={onset_delta} | min_gap={min_gap}s\n")

    examples      = {}
    total_found   = 0
    total_kept    = 0
    total_dropped = 0

    for track_file in track_files:
        # Nome "pulito" della traccia (senza estensione, senza spazi)
        track_name = track_file.stem.replace(" ", "_")

        print(f"─── {track_file.name}")

        # Carica audio
        try:
            waveform = load_audio(track_file)
        except Exception as e:
            print(f"    ERRORE caricamento: {e} — traccia saltata")
            continue

        duration_sec = len(waveform) / SAMPLE_RATE
        print(f"    Durata: {duration_sec:.1f}s  ({len(waveform):,} campioni)")

        # Onset detection
        onsets = detect_onsets(waveform, delta=onset_delta, min_gap=min_gap)
        print(f"    Onset trovati: {len(onsets)}")

        kept    = 0
        dropped = 0

        for i, onset_time in enumerate(onsets):
            total_found += 1

            segment = extract_segment(waveform, onset_time)
            rms     = compute_rms(segment)

            # Filtra silenziosi
            if rms < rms_threshold:
                dropped      += 1
                total_dropped += 1
                continue

            # Nome del segmento: traccia_onset_NNNN
            seg_name = f"{track_name}_onset_{i:04d}"

            # Salva .wav
            out_wav = audio_out / f"{seg_name}.wav"
            sf.write(str(out_wav), segment, SAMPLE_RATE, subtype="PCM_16")

            examples[seg_name] = {
                "source_track":          track_name,
                "onset_time":            round(onset_time, 3),
                "segment_index":         i,
                "rms":                   round(rms, 4),
                "instrument_family_str": track_name,   # compatibilità con dataset.py
                "pitch":                 0,             # non applicabile
            }

            kept        += 1
            total_kept  += 1

        print(f"    Mantenuti: {kept}  |  Scartati (troppo silenziosi): {dropped}")

    # Salva examples.json
    json_path = output_path / "examples.json"
    with open(json_path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"\n{'═'*50}")
    print(f"RIEPILOGO")
    print(f"  Tracce processate : {len(track_files)}")
    print(f"  Onset totali      : {total_found}")
    print(f"  Segmenti mantenuti: {total_kept}")
    print(f"  Segmenti scartati : {total_dropped}")
    print(f"  Output            : {output_dir}/")
    print(f"    ├── audio/      ({total_kept} file .wav)")
    print(f"    └── examples.json")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Costruisce dataset da tracce audio personali con onset detection"
    )
    parser.add_argument("--input_dir",     default=DEFAULT_INPUT_DIR,
                        help=f"Cartella con le tracce originali (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output_dir",    default=DEFAULT_OUTPUT_DIR,
                        help=f"Cartella di output (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--rms_threshold", type=float, default=DEFAULT_RMS_THRESH,
                        help=f"Soglia RMS minima (default: {DEFAULT_RMS_THRESH})")
    parser.add_argument("--onset_delta",   type=float, default=DEFAULT_ONSET_DELTA,
                        help=f"Sensibilità onset detection (default: {DEFAULT_ONSET_DELTA})")
    parser.add_argument("--min_gap",       type=float, default=DEFAULT_MIN_GAP,
                        help=f"Gap minimo tra onset in secondi (default: {DEFAULT_MIN_GAP})")
    args = parser.parse_args()

    build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rms_threshold=args.rms_threshold,
        onset_delta=args.onset_delta,
        min_gap=args.min_gap,
    )

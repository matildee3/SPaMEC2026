import os
import json
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


SAMPLE_RATE = 44100
DURATION    = 1.0
N_MELS      = 100
N_FFT       = 2048
HOP_LENGTH  = 689 


def audio_to_spectrogram(waveform, sr=SAMPLE_RATE):
    """
    Waveform → mel-spettrogramma in scala log.
    Ritorna array (100, 64) NON normalizzato.
    """
    target = int(sr * DURATION)
    if len(waveform) > target:
        waveform = waveform[:target]
    else:
        waveform = np.pad(waveform, (0, target - len(waveform)))

    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_log = np.log(mel + 1e-5)

    if mel_log.shape[1] != 64:
        mel_log = mel_log[:, :64] if mel_log.shape[1] > 64 else \
                  np.pad(mel_log, ((0,0),(0, 64 - mel_log.shape[1])))

    return mel_log.astype(np.float32)


def spectrogram_to_audio(spec_norm, mean, std, sr=SAMPLE_RATE):
    """
    Spettrogramma normalizzato → waveform audio.
    """
    mel_log = spec_norm * std + mean
    mel     = np.exp(mel_log) - 1e-5
    mel     = np.maximum(mel, 0)

    p99 = np.percentile(mel, 99)
    if p99 > 0:
        mel = np.clip(mel, 0, p99 * 2)

    waveform = librosa.feature.inverse.mel_to_audio(
        mel, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=128, power=1.0
    )

    mx = np.abs(waveform).max()
    if mx > 0:
        waveform = waveform / mx

    return waveform.astype(np.float32)


class NSynthDataset(Dataset):
    """
    Dataset generico per segmenti audio con examples.json.
    """
    def __init__(self, data_dir, max_samples=None, cache=True,
                 mean=None, std=None):
        self.data_dir  = Path(data_dir)
        self.audio_dir = self.data_dir / "audio"

        with open(self.data_dir / "examples.json") as f:
            self.metadata = json.load(f)

        self.filenames = sorted(self.metadata.keys())

        import random
        random.seed(42)
        random.shuffle(self.filenames)
        if max_samples:
            self.filenames = self.filenames[:max_samples]

        print(f"Dataset: {len(self.filenames)} segmenti da '{data_dir}'")

        track_counts = {}
        for fname in self.filenames:
            track = self.metadata[fname].get('source_track',
                    self.metadata[fname].get('instrument_family_str', 'unknown'))
            track_counts[track] = track_counts.get(track, 0) + 1
        print("  Distribuzione per traccia:")
        for track, count in sorted(track_counts.items()):
            bar = "█" * (count // 5)
            print(f"    {track:<30} {count:>4} segmenti  {bar}")

        # Cache spettrogrammi
        self._cache = {}
        if cache:
            print("Pre-processing spettrogrammi...")
            for i, fname in enumerate(self.filenames):
                self._cache[fname] = self._load_spec(fname)
                if (i+1) % 100 == 0:
                    print(f"  {i+1}/{len(self.filenames)}")
            print("Cache completata.")

        # Calcola o usa mean/std forniti
        if mean is None or std is None:
            all_specs  = np.stack([self._get_spec(f) for f in self.filenames])
            self.mean  = float(all_specs.mean())
            self.std   = float(all_specs.std()) + 1e-8
            print(f"Normalizzazione — mean: {self.mean:.3f}, std: {self.std:.3f}")
        else:
            self.mean = mean
            self.std  = std

    def _load_spec(self, filename):
        wav_path = self.audio_dir / f"{filename}.wav"
        waveform, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        return audio_to_spectrogram(waveform)

    def _get_spec(self, filename):
        return self._cache[filename] if filename in self._cache \
               else self._load_spec(filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname     = self.filenames[idx]
        spec      = self._get_spec(fname)
        spec_norm = (spec - self.mean) / self.std

        meta = self.metadata[fname]

    
        instrument = meta.get('instrument_family_str',
                     meta.get('source_track', 'unknown'))

        return {
            'spectrogram': torch.tensor(spec_norm).unsqueeze(0),  # (1, 80, 64)
            'filename':    fname,
            'instrument':  instrument,   # nome traccia nel dataset personale
            'pitch':       meta.get('pitch', 0),
            'source_track':  meta.get('source_track', instrument),
            'onset_time':    meta.get('onset_time', 0.0),
        }


def get_dataloader(data_dir, batch_size=32, max_samples=None,
                   shuffle=True, mean=None, std=None):
    dataset = NSynthDataset(data_dir, max_samples=max_samples,
                            mean=mean, std=std)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=0, drop_last=True)
    return loader, dataset




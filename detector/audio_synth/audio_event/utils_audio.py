#!/usr/bin/env python3
import numpy as np
import soundfile as sf
import librosa

def load_wav(path, target_sr=None):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if (target_sr is not None) and (sr != target_sr):
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr

def slice_samples(y, sr, t0, t1, pad=True):
    """Return samples in [t0, t1). If signal is short, pad zeros to exact length."""
    s0 = int(round(t0 * sr))
    s1 = int(round(t1 * sr))
    s0 = max(0, s0)
    s1 = max(s0, s1)
    seg = y[s0:s1]
    want = s1 - s0
    if pad and len(seg) < want:
        pad_len = want - len(seg)
        seg = np.pad(seg, (0, pad_len), mode="constant")
    return seg.astype(np.float32)

def to_logmel(
    y, sr, n_fft=1024, win_length=1024, hop_length=160,
    n_mels=64, fmin=20, fmax=None, top_db=80.0
):
    spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0, center=True
    )
    logmel = librosa.power_to_db(spec, ref=1.0, top_db=top_db).astype(np.float32)
    # per-window CMVN
    m = logmel.mean()
    s = logmel.std() + 1e-6
    logmel = (logmel - m) / s
    return logmel  # (n_mels, T)

def random_gain(y, low=0.8, high=1.2):
    g = np.random.uniform(low, high)
    return (y * g).astype(np.float32)

def add_gaussian_noise(y, snr_db_range=(10, 30)):
    # compute current rms
    rms = np.sqrt(np.mean(y ** 2) + 1e-12)
    snr_db = np.random.uniform(*snr_db_range)
    noise_rms = rms / (10 ** (snr_db / 20.0))
    n = np.random.randn(len(y)).astype(np.float32)
    n *= noise_rms / (np.sqrt(np.mean(n ** 2)) + 1e-12)
    return (y + n).astype(np.float32)

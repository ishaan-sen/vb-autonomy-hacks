#!/usr/bin/env python3
import csv, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils_audio import load_wav, slice_samples, to_logmel, random_gain, add_gaussian_noise

class SegmentsCSV:
    """Lightweight CSV reader that filters by split and holds rows."""
    def __init__(self, csv_path: Path, split: str):
        self.rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            # expect columns: split, street, path, seg_t0, seg_t1, label
            required = {"split", "street", "path", "seg_t0", "seg_t1", "label"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"segments.csv missing columns: {missing}")
            for r in reader:
                if r["split"].strip().lower() != split.lower():
                    continue
                self.rows.append(r)
        if len(self.rows) == 0:
            raise FileNotFoundError(f"No rows found for split='{split}' in {csv_path}")

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

class AudioSegDataset(Dataset):
    def __init__(self, csv_path: Path, split: str, cfg, augment: bool=False):
        self.meta = SegmentsCSV(csv_path, split)
        self.cfg = cfg
        self.augment = augment

        # compute pos/neg for pos_weight
        pos = sum(int(r["label"]) for r in self.meta.rows)
        neg = len(self.meta) - pos
        self.pos_count = pos
        self.neg_count = neg

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        r = self.meta[idx]
        path = Path(r["path"])
        t0 = float(r["seg_t0"]); t1 = float(r["seg_t1"])
        label = int(r["label"])

        y, sr = load_wav(path, target_sr=self.cfg.sample_rate)
        seg = slice_samples(y, sr, t0, t1, pad=True)

        # simple augmentation for training
        if self.augment:
            if random.random() < 0.5:
                seg = random_gain(seg, 0.9, 1.1)
            if random.random() < 0.5:
                seg = add_gaussian_noise(seg, (15, 30))

        m = to_logmel(
            seg, sr, n_fft=self.cfg.n_fft, win_length=self.cfg.win_length,
            hop_length=self.cfg.hop_length, n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin, fmax=self.cfg.fmax, top_db=self.cfg.top_db
        )  # (F, T)

        # to torch: (1, F, T)
        x = torch.from_numpy(m).unsqueeze(0)  # C=1
        y = torch.tensor([label], dtype=torch.float32)
        return x, y, path.as_posix(), t0, t1

def collate_fn(batch):
    xs, ys, paths, t0s, t1s = zip(*batch)
    # pad to same time dim if slightly different
    max_T = max(x.shape[-1] for x in xs)
    xs_pad = []
    for x in xs:
        if x.shape[-1] < max_T:
            pad = torch.zeros((x.shape[0], x.shape[1], max_T - x.shape[-1]), dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        xs_pad.append(x)
    X = torch.stack(xs_pad, dim=0)  # (B,1,F,T)
    Y = torch.stack(ys, dim=0)      # (B,1)
    return X, Y.squeeze(1), paths, t0s, t1s

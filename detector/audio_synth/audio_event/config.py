#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

class Cfg:
    # Data
    data_root = Path("/home/UFAD/zhou.zhuoyang/hacks/AIDER/audio_synth/audio_dataset")  # change if needed
    segments_csv = "segments.csv"
    sample_rate = 16000

    # Feature (Log-Mel)
    n_fft = 1024
    win_length = 1024
    hop_length = 160          # 10 ms @ 16k
    n_mels = 64
    fmin = 20
    fmax = 8000
    top_db = 80.0

    # Training
    batch_size = 256
    num_workers = 32
    epochs = 40
    lr = 1e-3
    weight_decay = 1e-4
    amp = True                # mixed precision
    seed = 2025

    # Checkpoints / logs
    ckpt_dir = Path("/home/UFAD/zhou.zhuoyang/hacks/AIDER/audio_synth/audio_event/ckpts")
    best_name = "best.pt"

    # Inference / post-process
    open_th = 0.6
    close_th = 0.4
    min_event_sec = 0.15
    smooth_win = 3            # moving average over segment probs

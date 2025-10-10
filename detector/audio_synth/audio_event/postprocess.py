#!/usr/bin/env python3
from typing import List, Tuple
import numpy as np

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    c = np.convolve(xp, np.ones(k)/k, mode="valid")
    return c.astype(np.float32)

def hysteresis(pred: np.ndarray, open_th=0.6, close_th=0.4) -> List[Tuple[int,int]]:
    """Return list of (start_idx, end_idx) on a binary sequence obtained by hysteresis."""
    events = []
    active = False
    s = 0
    for i, p in enumerate(pred):
        if not active and p >= open_th:
            active = True
            s = i
        elif active and p < close_th:
            active = False
            events.append((s, i))
    if active:
        events.append((s, len(pred)))
    return events

def merge_close_events(events: List[Tuple[int,int]], min_gap=1) -> List[Tuple[int,int]]:
    if not events: return []
    merged = [events[0]]
    for s,e in events[1:]:
        ps,pe = merged[-1]
        if s - pe <= min_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s,e))
    return merged

def remove_short(events: List[Tuple[int,int]], min_len=1) -> List[Tuple[int,int]]:
    return [(s,e) for (s,e) in events if (e - s) >= min_len]

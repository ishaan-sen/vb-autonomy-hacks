#!/usr/bin/env python3
import csv, argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from .config import Cfg
from .dataset import AudioSegDataset, collate_fn
from .models.crnn import CRNNEvent
from .postprocess import moving_average, hysteresis, merge_close_events, remove_short


def parse_street_id(clip_path: str) -> str:
    """
    Robustly parse street id from filename like:
      ".../street_07_train_023.wav" -> "street_07"
      ".../street_07_val_010.wav"   -> "street_07"
    If your naming differs, adjust here.
    """
    name = Path(clip_path).name  # e.g., street_07_train_023.wav
    parts = name.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return parts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root folder that contains wave/ and segments.csv")
    parser.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    parser.add_argument("--ckpt", type=str, required=True, help="Path to event detector checkpoint (.pt)")
    parser.add_argument("--open_th", type=float, default=None, help="Override open threshold for hysteresis")
    parser.add_argument("--close_th", type=float, default=None, help="Override close threshold for hysteresis")
    parser.add_argument("--min_event_sec", type=float, default=None, help="Minimum event length in seconds")
    parser.add_argument("--smooth_win", type=int, default=None, help="Moving-average window (segments)")
    parser.add_argument("--out_prefix", type=str, default="./detections",
                        help="Prefix for output files; will create *_segments.csv, *_events.csv, *_streets.csv, *_meta.json")
    parser.add_argument("--write_zero_event_streets", action="store_true",
                        help="Also write streets with zero detected events in *_streets.csv")
    args = parser.parse_args()

    # ----------------------
    # Config
    # ----------------------
    cfg = Cfg()
    if args.data_root is not None:
        cfg.data_root = Path(args.data_root)
    if args.open_th is not None:     cfg.open_th = args.open_th
    if args.close_th is not None:    cfg.close_th = args.close_th
    if args.min_event_sec is not None: cfg.min_event_sec = args.min_event_sec
    if args.smooth_win is not None:  cfg.smooth_win = args.smooth_win

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------
    # Dataset / Loader (NO AUG in test)
    # ----------------------
    seg_csv = cfg.data_root / cfg.segments_csv
    ds = AudioSegDataset(seg_csv, args.split, cfg, augment=False)
    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # ----------------------
    # Model
    # ----------------------
    model = CRNNEvent(n_mels=cfg.n_mels).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    saved_default_th = float(ckpt.get("thresh", cfg.open_th))

    # ----------------------
    # Collect raw segment probabilities per clip
    # ----------------------
    # clip -> list of (t0, t1, prob)
    clip_probs = defaultdict(list)
    # keep also street list for zero-event reporting
    all_streets = set()

    with torch.no_grad():
        for X, Y, paths, t0s, t1s in tqdm(dl, desc="[Infer]", ncols=100):
            X = X.to(device, non_blocking=True)
            logits = model(X)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

            for pth, t0, t1, pr in zip(paths, t0s, t1s, probs):
                clip_probs[pth].append((float(t0), float(t1), float(pr)))
                all_streets.add(parse_street_id(pth))

    # ----------------------
    # Write per-segment CSV (raw)
    # ----------------------
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    seg_csv_out = out_prefix.with_suffix("")  # strip suffix if any
    seg_csv_out = Path(str(seg_csv_out) + "_segments.csv")
    with open(seg_csv_out, "w", newline="") as fseg:
        wseg = csv.writer(fseg)
        wseg.writerow(["clip", "street", "seg_t0", "seg_t1", "prob"])
        for clip_path, rows in clip_probs.items():
            street_id = parse_street_id(clip_path)
            # sort by seg start time
            rows_sorted = sorted(rows, key=lambda x: x[0])
            for (t0, t1, pr) in rows_sorted:
                wseg.writerow([clip_path, street_id, f"{t0:.3f}", f"{t1:.3f}", f"{pr:.6f}"])

    # ----------------------
    # Post-process into events per clip
    # ----------------------
    # street -> list of event dicts
    street_events = defaultdict(list)

    events_csv_out = out_prefix.with_suffix("")
    events_csv_out = Path(str(events_csv_out) + "_events.csv")
    with open(events_csv_out, "w", newline="") as fevt:
        wevt = csv.writer(fevt)
        wevt.writerow([
            "street", "clip", "event_idx", "t_start", "t_end",
            "len_sec", "max_prob", "mean_prob", "open_th", "close_th", "smooth_win"
        ])

        for clip_path, rows in clip_probs.items():
            rows.sort(key=lambda x: x[0])  # sort by t0
            t0s = np.array([r[0] for r in rows], dtype=np.float32)
            t1s = np.array([r[1] for r in rows], dtype=np.float32)
            probs = np.array([r[2] for r in rows], dtype=np.float32)

            # smooth
            probs_s = moving_average(probs, cfg.smooth_win)

            # segment hop (sec)
            hop_sec = max(1e-6, np.median(t1s - t0s))
            min_len_seg = max(1, int(round(cfg.min_event_sec / hop_sec)))

            # hysteresis on smoothed probs
            evs = hysteresis(probs_s, open_th=cfg.open_th, close_th=cfg.close_th)  # list of (s, e) in segment indices
            # optional morph ops
            evs = merge_close_events(evs, min_gap=1)
            evs = remove_short(evs, min_len=min_len_seg)

            street_id = parse_street_id(clip_path)
            for k, (s, e) in enumerate(evs):
                # convert to seconds using the segment timeline
                t_start = float(t0s[s])
                t_end   = float(t1s[min(e-1, len(t1s)-1)])
                pmax    = float(probs_s[s:e].max()) if e > s else float(probs_s[s])
                pmean   = float(probs_s[s:e].mean()) if e > s else float(probs_s[s])
                ev_len  = max(0.0, t_end - t_start)

                evrec = {
                    "clip": clip_path, "t_start": t_start, "t_end": t_end,
                    "max_prob": pmax, "mean_prob": pmean, "len_sec": ev_len
                }
                street_events[street_id].append(evrec)

                wevt.writerow([
                    street_id, clip_path, k, f"{t_start:.3f}", f"{t_end:.3f}",
                    f"{ev_len:.3f}", f"{pmax:.6f}", f"{pmean:.6f}",
                    f"{cfg.open_th:.4f}", f"{cfg.close_th:.4f}", cfg.smooth_win
                ])

    # ----------------------
    # Aggregate per-street
    # ----------------------
    streets_csv_out = out_prefix.with_suffix("")
    streets_csv_out = Path(str(streets_csv_out) + "_streets.csv")
    with open(streets_csv_out, "w", newline="") as fs:
        ws = csv.writer(fs)
        ws.writerow(["street", "n_events", "total_event_sec", "max_prob"])

        # if requested, include zero-event streets
        streets_iter = sorted(all_streets) if args.write_zero_event_streets \
                       else sorted(street_events.keys())

        for street in streets_iter:
            evs = street_events.get(street, [])
            if len(evs) == 0 and not args.write_zero_event_streets:
                continue
            n_events = len(evs)
            total_sec = sum(ev["t_end"] - ev["t_start"] for ev in evs) if n_events > 0 else 0.0
            max_p = max((ev["max_prob"] for ev in evs), default=0.0)
            ws.writerow([street, n_events, f"{total_sec:.3f}", f"{max_p:.6f}"])

    # ----------------------
    # Meta info
    # ----------------------
    meta_out = out_prefix.with_suffix("")
    meta_out = Path(str(meta_out) + "_meta.json")
    meta = {
        "split": args.split,
        "data_root": str(cfg.data_root),
        "ckpt": args.ckpt,
        "threshold_saved_in_ckpt": saved_default_th,
        "open_th_used": cfg.open_th,
        "close_th_used": cfg.close_th,
        "smooth_win_used": cfg.smooth_win,
        "min_event_sec_used": cfg.min_event_sec,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "n_mels": cfg.n_mels,
        "segments_csv_path": str(seg_csv),
        "outputs": {
            "segments_csv": str(seg_csv_out),
            "events_csv": str(events_csv_out),
            "streets_csv": str(streets_csv_out),
            "meta_json": str(meta_out),
        }
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2))

    print("[DONE]")
    print(f"  segments -> {seg_csv_out}")
    print(f"  events   -> {events_csv_out}")
    print(f"  streets  -> {streets_csv_out}")
    print(f"  meta     -> {meta_out}")


if __name__ == "__main__":
    main()

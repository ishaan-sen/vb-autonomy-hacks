#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic audio dataset generator (street-level normal vs anomaly).
Each street is either fully normal (all clips have no jump events)
or fully anomalous (all clips contain jump events).
"""

import os, math, csv, json, random, argparse
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write as wavwrite
from tqdm import tqdm

# -----------------------------
# Default parameters
# -----------------------------
DEFAULT_SR = 16000
DEFAULT_CLIP_SEC = 20
DEFAULT_N_STREETS = 24
DEFAULT_TRAIN_PER_STREET = 60
DEFAULT_VAL_PER_STREET = 16
DEFAULT_TEST_PER_STREET = 16
DEFAULT_ANOM_RATE = 0.2  # proportion of streets that are anomalous
RANDOM_SEED = 2025
SEG_WIN = 1.5
SEG_HOP = 0.5

ANOM_EVENT_CFG = {
    "n_events_poisson_lambda": 1.2,
    "min_dur": 0.05, "max_dur": 0.35,
    "snr_db_range": (6, 24),
    "types": ["gunshot", "impact"]
}

NON_ANOM_EVENT_CFG = {
    "enable": True,
    "n_events_poisson_lambda": 1.0,
    "min_dur": 0.2, "max_dur": 0.8,
    "snr_db_range": (-8, 6),
    "types": ["car_pass", "dog_bark_soft", "horn_soft"]
}

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed); random.seed(seed)

def db2lin(db): return 10 ** (db / 20)
def clip_norm(x): return np.clip(x, -1.0, 1.0).astype(np.float32)

def pink_noise(n, sr=DEFAULT_SR, power=1.0):
    X = np.fft.rfft(np.random.randn(n))
    freqs = np.fft.rfftfreq(n, 1/sr)
    freqs[0] = 1.0
    X /= (freqs ** 0.5)
    y = np.fft.irfft(X, n)
    y = y / (np.std(y)+1e-9)
    return (y*power).astype(np.float32)

def sine_tone(n, sr, f0, amp=0.1, pm_depth=0.0, fm_depth=0.0):
    t = np.arange(n)/sr
    phase = 2*np.pi*f0*t
    if fm_depth > 0:
        f_inst = f0 * (1 + fm_depth*np.sin(2*np.pi*0.2*t))
        phase = 2*np.pi*np.cumsum(f_inst)/sr
    if pm_depth > 0:
        phase = phase + pm_depth*np.sin(2*np.pi*0.5*t)
    return (np.sin(phase)*amp).astype(np.float32)

def apply_envelope(x, attack=0.005, release=0.02, sr=DEFAULT_SR):
    n = len(x)
    if n == 0:
        return x
    y = x.astype(np.float32, copy=True)
    a = int(round(attack * sr))
    r = int(round(release * sr))
    a = max(0, min(a, n))
    r = max(0, min(r, n))
    if a > 0:
        y[:a] *= np.linspace(0.0, 1.0, num=a, dtype=np.float32)
    if r > 0:
        y[n - r:] *= np.linspace(1.0, 0.0, num=r, dtype=np.float32)
    return y

def synth_event(ev_type, dur_s, sr=DEFAULT_SR):
    n = int(dur_s*sr); t = np.arange(n)/sr
    if ev_type=="gunshot":
        burst = np.random.randn(n).astype(np.float32)
        decay = np.exp(-t*40.0).astype(np.float32)
        y = burst*decay; y = apply_envelope(y,0.003,0.05,sr)
    elif ev_type=="impact":
        y = 0.7*sine_tone(n,sr,np.random.uniform(120,220),amp=1.0) \
          + 0.3*np.random.randn(n).astype(np.float32)
        y*=np.exp(-t*12.0).astype(np.float32); y=apply_envelope(y,0.004,0.1,sr)
    elif ev_type=="car_pass":
        base = pink_noise(n,sr,power=0.4)
        doppler = np.sin(2*np.pi*0.25*t)
        y = base*(0.5+0.5*doppler); y=apply_envelope(y,0.2,0.2,sr)
    elif ev_type=="dog_bark_soft":
        y = np.sign(np.sin(2*np.pi*np.random.uniform(300,500)*t))*0.2
        y*=np.exp(-t*6.0).astype(np.float32); y=apply_envelope(y,0.01,0.2,sr)
    elif ev_type=="horn_soft":
        y = sine_tone(n,sr,np.random.uniform(400,600),amp=0.2,pm_depth=0.2)
        y=apply_envelope(y,0.02,0.2,sr)
    else: y=np.zeros(n,dtype=np.float32)
    return y/(np.max(np.abs(y))+1e-9)

def make_street_profiles(n_streets, anom_rate):
    profiles=[]
    n_anom = int(round(n_streets*anom_rate))
    anom_ids = set(random.sample(range(n_streets), n_anom))
    for i in range(n_streets):
        profiles.append({
            "name": f"street_{i:02d}",
            "base_f0": np.random.uniform(40,250),
            "pink_pow": np.random.uniform(0.05,0.25),
            "fm_depth": np.random.uniform(0.00,0.10),
            "tone_amp": np.random.uniform(0.02,0.10),
            "is_anom_street": (i in anom_ids)
        })
    return profiles

def synth_ambient(profile, sec, sr=DEFAULT_SR):
    n=int(sec*sr)
    amb=pink_noise(n,sr=sr,power=profile["pink_pow"])
    tone=sine_tone(n,sr,f0=profile["base_f0"],amp=profile["tone_amp"],fm_depth=profile["fm_depth"])
    t=np.arange(n)/sr
    gain=0.9+0.2*np.sin(2*np.pi*np.random.uniform(0.02,0.05)*t+np.random.uniform(0,2*np.pi))
    y=amb+tone*gain.astype(np.float32)
    return (y/(np.max(np.abs(y))+1e-9)*0.6).astype(np.float32)

def mix_event_to_bg(bg, ev, start_idx, snr_db):
    seg=bg[start_idx:start_idx+len(ev)]
    bg_rms=np.sqrt(np.mean(seg**2)+1e-12)
    ev_rms=np.sqrt(np.mean(ev**2)+1e-12)
    target_ev_rms=bg_rms*db2lin(snr_db)
    ev_scaled=ev*(target_ev_rms/(ev_rms+1e-12))
    bg[start_idx:start_idx+len(ev)]+=ev_scaled
    return bg

def synth_clip(profile, sr=DEFAULT_SR):
    n=int(DEFAULT_CLIP_SEC*sr)
    y=synth_ambient(profile,DEFAULT_CLIP_SEC,sr)
    events=[]
    # always allow non-anomaly events
    if NON_ANOM_EVENT_CFG["enable"]:
        k=np.random.poisson(NON_ANOM_EVENT_CFG["n_events_poisson_lambda"])
        for _ in range(k):
            dur=np.random.uniform(NON_ANOM_EVENT_CFG["min_dur"],NON_ANOM_EVENT_CFG["max_dur"])
            ev=synth_event(random.choice(NON_ANOM_EVENT_CFG["types"]),dur,sr)
            start=np.random.randint(0,max(1,n-len(ev)-1))
            snr_db=np.random.uniform(*NON_ANOM_EVENT_CFG["snr_db_range"])
            y=mix_event_to_bg(y,ev,start,snr_db)
    # anomaly street: inject jump events
    if profile["is_anom_street"]:
        k=max(1,np.random.poisson(ANOM_EVENT_CFG["n_events_poisson_lambda"]))
        for _ in range(k):
            dur=np.random.uniform(ANOM_EVENT_CFG["min_dur"],ANOM_EVENT_CFG["max_dur"])
            ev=synth_event(random.choice(ANOM_EVENT_CFG["types"]),dur,sr)
            start=np.random.randint(0,max(1,n-len(ev)-1))
            snr_db=np.random.uniform(*ANOM_EVENT_CFG["snr_db_range"])
            y=mix_event_to_bg(y,ev,start,snr_db)
            events.append({"t_start":start/sr,"t_end":(start+len(ev))/sr,"type":"jump"})
    return clip_norm(y), events

def slice_segments(duration, events, win=SEG_WIN, hop=SEG_HOP):
    segs=[]; t=0.0
    while t+win<=duration+1e-6:
        t0,t1=t,t+win; label=0
        for ev in events:
            if not (t1<=ev["t_start"] or t0>=ev["t_end"]): label=1; break
        segs.append((t0,t1,label)); t+=hop
    return segs

# -----------------------------
# Generate one split
# -----------------------------
def generate_split(split, per_street, profiles, wave_dir, events_w, seg_w, args):
    for prof in tqdm(profiles, desc=f"[{split}] streets", ncols=100):
        for i in range(per_street):
            y,events=synth_clip(prof,args.sr)
            subdir="anomaly" if prof["is_anom_street"] else "normal"
            fname=f"{prof['name']}_{split}_{i:03d}.wav"
            fpath=wave_dir/f"{split}/{subdir}/{fname}"
            wavwrite(fpath.as_posix(), args.sr, (y*32767).astype(np.int16))
            for ev in events:
                events_w.writerow([split,prof["name"],fpath.as_posix(),ev["t_start"],ev["t_end"],ev["type"],int(prof["is_anom_street"])])
            segs=slice_segments(args.clip_sec,events)
            for (t0,t1,label) in segs:
                seg_w.writerow([split,prof["name"],fpath.as_posix(),t0,t1,label])

# -----------------------------
# Main
# -----------------------------
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=str,default="./audio_dataset")
    parser.add_argument("--sr",type=int,default=DEFAULT_SR)
    parser.add_argument("--clip-sec",type=float,default=DEFAULT_CLIP_SEC)
    parser.add_argument("--streets",type=int,default=DEFAULT_N_STREETS)
    parser.add_argument("--train-per-street",type=int,default=DEFAULT_TRAIN_PER_STREET)
    parser.add_argument("--val-per-street",type=int,default=DEFAULT_VAL_PER_STREET)
    parser.add_argument("--test-per-street",type=int,default=DEFAULT_TEST_PER_STREET)
    parser.add_argument("--anom-rate",type=float,default=DEFAULT_ANOM_RATE, help="proportion of streets that are anomalous")
    args=parser.parse_args()

    set_seed(RANDOM_SEED)
    outdir=Path(args.out); wave_dir=outdir/"wave"
    for sub in ["train/normal","train/anomaly","val/normal","val/anomaly","test/normal","test/anomaly"]:
        (wave_dir/sub).mkdir(parents=True,exist_ok=True)
    print(f"[INFO] Output root: {outdir}")

    profiles=make_street_profiles(args.streets, args.anom_rate)
    (outdir/"meta.json").write_text(json.dumps(vars(args),indent=2))
    events_csv=open(outdir/"events.csv","w",newline="")
    segments_csv=open(outdir/"segments.csv","w",newline="")
    events_w=csv.writer(events_csv); seg_w=csv.writer(segments_csv)
    events_w.writerow(["split","street","path","t_start","t_end","type","is_anomaly_street"])
    seg_w.writerow(["split","street","path","seg_t0","seg_t1","label"])
    generate_split("train",args.train_per_street,profiles,wave_dir,events_w,seg_w,args)
    generate_split("val",args.val_per_street,profiles,wave_dir,events_w,seg_w,args)
    generate_split("test",args.test_per_street,profiles,wave_dir,events_w,seg_w,args)
    events_csv.close(); segments_csv.close()
    print("[DONE] Dataset generated.")

if __name__=="__main__":
    main()

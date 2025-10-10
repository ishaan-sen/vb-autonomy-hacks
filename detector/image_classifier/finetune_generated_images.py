#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune on a subset of the generated_images dataset (3 classes: normal / waterlogging / traffic_incident)
while loading the original 5-class classifier checkpoint. We train/eval using only the 3 relevant logits
by slicing the model outputs, so no head surgery is required.

Data root is hard-coded:
  /home/UFAD/zhou.zhuoyang/hacks/AIDER/data/generated_images

Split (stratified):
  train_ft = 20%, val_ft = 10%, test_holdout = 70%

Outputs:
  - best fine-tuned ckpt:   generated_images/ckpts_ft/best.pt
  - per-image predictions:  generated_images/ckpts_ft/test_preds.csv
  - confusion matrix .npy:  generated_images/ckpts_ft/cm_test.npy
"""

import os
from pathlib import Path
import random
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from config import CFG
from models.classifier import build_model
from utils.common import EMA

# -----------------------------
# Hard-coded paths & classes
# -----------------------------
DATA_ROOT = Path("/home/UFAD/zhou.zhuoyang/hacks/AIDER/data/generated_images")
OUT_DIR   = DATA_ROOT / "ckpts_ft"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Folder name -> model's 5-class label name
FOLDER_TO_MODEL = {
    "normal": "normal",
    "waterlogging": "flooded_areas",
    "flooded_areas": "flooded_areas",
    "traffic_incident": "traffic_incident",
}
REPORT_NAMES = ["normal", "waterlogging", "traffic_incident"]  # 3-class names used in reports

SEED = 2025
TRAIN_RATIO = 0.40
VAL_RATIO   = 0.10  # test = 1 - (train+val) = 0.70
BATCH_SIZE = 64
EPOCHS = 20
LR = 5e-5
WD = 1e-4
AMP = True
PATIENCE = 5


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def collect_images(root: Path):
    paths, labels = [], []
    for folder in FOLDER_TO_MODEL.keys():
        d = root / folder
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                paths.append(p)
                # map "waterlogging"->"waterlogging" (later we map to flooded_areas index)
                if folder in ("waterlogging", "flooded_areas"):
                    labels.append("waterlogging")
                else:
                    labels.append(folder)
    return np.array(paths), np.array(labels)


class ImgDS(Dataset):
    def __init__(self, paths, labels, tfm):
        self.paths = list(paths)
        self.labels = list(labels)
        self.tfm = tfm
        self.name_to_id = {n: i for i, n in enumerate(REPORT_NAMES)}

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        y = self.name_to_id[self.labels[i]]
        img = Image.open(p).convert("RGB")
        x = self.tfm(img)
        return x, y, str(p)


def build_tfms(train=True, img_size=256):
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.3),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])


def load_model_and_indices(device):
    cfg = CFG()
    model = build_model(cfg.model_name, num_classes=len(cfg.classes)).to(device)
    ckpt = torch.load(cfg.out_dir() / cfg.ckpt_best, map_location=device)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("ema"):
        ema_obj = EMA(model)
        ema_obj.load_state_dict(ckpt["ema"])
        model = ema_obj.ema
    model.eval()

    idx_normal  = cfg.classes.index("normal")
    idx_flood   = cfg.classes.index("flooded_areas")
    idx_traffic = cfg.classes.index("traffic_incident")
    idx_sel = [idx_normal, idx_flood, idx_traffic]
    return model, idx_sel


def train_finetune(model, idx_sel, loaders, class_weights=None, device="cuda"):
    # We will fine-tune ALL layers with a low LR
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    best_f1, best_state, bad = -1.0, None, 0

    for epoch in range(1, EPOCHS+1):
        # ---- Train ----
        model.train()
        running = 0.0
        for x, y, _ in loaders["train"]:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=AMP):
                logits = model(x)              # (B,5)
                logits3 = logits[:, idx_sel]   # (B,3)
                loss = criterion(logits3, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item() * x.size(0)
        tr_loss = running / len(loaders["train"].dataset)

        # ---- Valid ----
        val_loss, y_true, y_pred = evaluate(model, idx_sel, loaders["val"], device, ret_loss=True, criterion=criterion)
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_macroF1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[EarlyStop] best val_macroF1={best_f1:.4f}")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        torch.save({"model": best_state}, OUT_DIR / "best.pt")
        print(f"** Saved best to {OUT_DIR / 'best.pt'}")
    return model


@torch.no_grad()
def evaluate(model, idx_sel, loader, device, ret_loss=False, criterion=None):
    model.eval()
    probs_all, y_all, paths = [], [], []
    running = 0.0
    for x, y, p in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        sel = logits[:, idx_sel]
        probs = torch.softmax(sel, dim=1)
        probs_all.append(probs.cpu().numpy())
        y_all.append(y.cpu().numpy())
        paths.extend(p)
        if ret_loss and criterion is not None:
            running += criterion(sel, y).item() * x.size(0)
    probs_all = np.concatenate(probs_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    y_pred = np.argmax(probs_all, axis=1)
    if ret_loss and criterion is not None:
        val_loss = running / len(loader.dataset)
        return val_loss, y_all, y_pred
    return y_all, y_pred, probs_all, paths


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Collect & stratified split
    paths, labels = collect_images(DATA_ROOT)
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under {DATA_ROOT}")
    print(f"[INFO] Total images: {len(paths)} | Class counts:",
          {c: int((labels == c).sum()) for c in REPORT_NAMES})

    # indices split: first get train+val (30%), then split val (其中 1/3 of 30% = 10%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - (TRAIN_RATIO + VAL_RATIO), random_state=SEED)
    trainval_idx, test_idx = next(sss1.split(paths, labels))
    paths_trv, labels_trv = paths[trainval_idx], labels[trainval_idx]
    paths_test, labels_test = paths[test_idx], labels[test_idx]

    rel_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)  # 1/3
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=SEED)
    train_idx, val_idx = next(sss2.split(paths_trv, labels_trv))
    paths_train, labels_train = paths_trv[train_idx], labels_trv[train_idx]
    paths_val, labels_val = paths_trv[val_idx], labels_trv[val_idx]

    print(f"[SPLIT] train={len(paths_train)}  val={len(paths_val)}  test={len(paths_test)}")

    # 2) Datasets & loaders
    ds_train = ImgDS(paths_train, labels_train, build_tfms(train=True))
    ds_val   = ImgDS(paths_val,   labels_val,   build_tfms(train=False))
    ds_test  = ImgDS(paths_test,  labels_test,  build_tfms(train=False))

    # class weights (inverse frequency on train)
    name_to_id = {n:i for i,n in enumerate(REPORT_NAMES)}
    counts = np.bincount([name_to_id[n] for n in labels_train], minlength=3)
    weights = (counts.sum() / (counts + 1e-9))
    weights = weights / weights.mean()
    print("[INFO] Train counts:", dict(zip(REPORT_NAMES, counts.tolist())))
    print("[INFO] Class weights (normed):", weights.round(3).tolist())

    loaders = {
        "train": DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True),
        "val":   DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
        "test":  DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
    }

    # 3) Load model & fine-tune on 3-class slice
    model, idx_sel = load_model_and_indices(device)
    model = train_finetune(model, idx_sel, loaders, class_weights=weights, device=device)

    # 4) Evaluate on hold-out test
    y_true, y_pred, probs, test_paths = evaluate(model, idx_sel, loaders["test"], device)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    print(f"\n=== HOLD-OUT TEST (never seen during fine-tune) ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix [rows=true, cols=pred]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=REPORT_NAMES, digits=4))

    # save per-image predictions
    out_csv = OUT_DIR / "test_preds.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","label_true","prob_normal","prob_waterlogging","prob_traffic_incident","pred_label"])
        for p, yt, pr in zip(test_paths, y_true, probs):
            w.writerow([p, REPORT_NAMES[int(yt)], f"{pr[0]:.6f}", f"{pr[1]:.6f}", f"{pr[2]:.6f}", REPORT_NAMES[int(np.argmax(pr))]])
    np.save(OUT_DIR / "cm_test.npy", cm)
    print(f"[INFO] Saved: {out_csv}")
    print(f"[INFO] Saved: {OUT_DIR / 'cm_test.npy'}")


if __name__ == "__main__":
    main()

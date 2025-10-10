#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate fixed dataset path with classes: normal / waterlogging / traffic_incident
using the existing 5-class classifier. 'waterlogging' is mapped to the model's
'flooded_areas'. Outputs metrics and per-image CSV.
"""

from pathlib import Path
from PIL import Image
import csv
import numpy as np
import torch
from torchvision import transforms as T
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import CFG
from models.classifier import build_model
from utils.common import EMA

ROOT_PATH = Path("/home/UFAD/zhou.zhuoyang/hacks/AIDER/data/generated_images")
OUT_CSV = ROOT_PATH / "preds_three_class.csv"

# Mapping: folder name in dataset -> model class name
FOLDER_TO_MODEL = {
    "normal": "normal",
    "waterlogging": "flooded_areas",
    "flooded_areas": "flooded_areas",
    "traffic_incident": "traffic_incident",
}

# Report labels order
REPORT_NAMES = ["normal", "flooded_areas", "traffic_incident"]


def load_model(device):
    cfg = CFG()
    model = build_model(cfg.model_name, num_classes=len(cfg.classes)).to(device)
    ckpt = torch.load(cfg.out_dir() / cfg.ckpt_best, map_location=device)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("ema"):
        ema_obj = EMA(model)
        ema_obj.load_state_dict(ckpt["ema"])
        model = ema_obj.ema
    model.eval()
    return model, cfg


def build_tfms(img_size=256):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def collect_images(root: Path):
    paths, labels = [], []
    for folder in FOLDER_TO_MODEL.keys():
        d = root / folder
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                paths.append(p)
                labels.append(folder)
    return paths, labels


def infer_batches(model, paths, tfms, device, batch_size, idx_sel, tta=False):
    probs_all = []
    with torch.no_grad():
        batch_imgs = []
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            ten = tfms(img)
            if tta:
                ten_flip = tfms(img.transpose(Image.FLIP_LEFT_RIGHT))
                ten = (ten + ten_flip) / 2.0
            batch_imgs.append(ten)

            if len(batch_imgs) == batch_size or i == len(paths) - 1:
                x = torch.stack(batch_imgs, dim=0).to(device)
                logits = model(x)
                sel = logits[:, idx_sel]  # 取对应的3类
                three = torch.softmax(sel, dim=1).cpu().numpy()
                probs_all.append(three)
                batch_imgs.clear()
    return np.concatenate(probs_all, axis=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths, folders = collect_images(ROOT_PATH)
    if not paths:
        raise FileNotFoundError(f"No images found under {ROOT_PATH}")
    print(f"[INFO] Found {len(paths)} images under {ROOT_PATH}")

    label_to_id = {n: i for i, n in enumerate(REPORT_NAMES)}
    y_true_names = []
    for f in folders:
        y_true_names.append("flooded_areas" if f in ("flooded_areas", "flooded_areas") else f)
    y_true = np.array([label_to_id[n] for n in y_true_names], dtype=np.int64)

    model, cfg = load_model(device)
    idx_normal = cfg.classes.index("normal")
    idx_flood = cfg.classes.index("flooded_areas")
    idx_traffic = cfg.classes.index("traffic_incident")
    idx_sel = [idx_normal, idx_flood, idx_traffic]

    tfms = build_tfms(256)
    probs = infer_batches(model, paths, tfms, device, batch_size=64, idx_sel=idx_sel, tta=True)
    y_pred = np.argmax(probs, axis=1)

    print("[INFO] Class counts:", {name: int((y_true == i).sum()) for i, name in enumerate(REPORT_NAMES)})
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print(f"\n=== Three-class evaluation ===\nAccuracy: {acc:.4f}")
    print("Confusion matrix [rows=true, cols=pred]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=REPORT_NAMES, digits=4))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label_true", "prob_normal", "prob_waterlogging", "prob_traffic_incident", "pred_label"])
        for p, yt, pr in zip(paths, y_true, probs):
            w.writerow([
                p.as_posix(),
                REPORT_NAMES[int(yt)],
                f"{pr[0]:.6f}", f"{pr[1]:.6f}", f"{pr[2]:.6f}",
                REPORT_NAMES[int(np.argmax(pr))]
            ])
    print(f"[INFO] Saved per-image predictions to: {OUT_CSV}")


if __name__ == "__main__":
    main()

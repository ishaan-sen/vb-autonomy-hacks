#!/usr/bin/env python3
import os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from audio_event.config import Cfg
from audio_event.dataset import AudioSegDataset, collate_fn
from audio_event.models.crnn import CRNNEvent
from audio_event.metrics import best_f1_from_probs

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    cfg = Cfg()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    seg_csv = cfg.data_root / cfg.segments_csv

    ds_tr = AudioSegDataset(seg_csv, "train", cfg, augment=True)
    ds_va = AudioSegDataset(seg_csv, "val", cfg, augment=False)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)

    model = CRNNEvent(n_mels=cfg.n_mels).to(device)

    # pos_weight for BCE
    pos = max(1, ds_tr.pos_count)
    neg = max(1, ds_tr.neg_count)
    pos_weight = torch.tensor([neg/pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)

    best_f1 = -1.0
    patience, bad = 8, 0

    for epoch in range(1, cfg.epochs+1):
        # ----------------- train -----------------
        model.train()
        losses=[]
        for X, Y, *_ in tqdm(dl_tr, desc=f"[Train] epoch {epoch}", ncols=100):
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg.amp):
                logits = model(X)
                loss = criterion(logits, Y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

        # ----------------- valid -----------------
        model.eval()
        probs_list=[]; labels_list=[]; val_losses=[]
        with torch.no_grad():
            for X, Y, *_ in tqdm(dl_va, desc="[Valid]", ncols=100):
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                logits = model(X)
                loss = criterion(logits, Y)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                labels = Y.detach().cpu().numpy()
                probs_list.append(probs); labels_list.append(labels)
        probs = np.concatenate(probs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        # metrics
        prauc = average_precision_score(labels, probs)
        rocauc = roc_auc_score(labels, probs)
        f1, th = best_f1_from_probs(labels, probs)

        print(f"Epoch {epoch:03d} | train_loss={np.mean(losses):.4f}  "
              f"val_loss={np.mean(val_losses):.4f}  PR-AUC={prauc:.4f} ROC-AUC={rocauc:.4f}  F1={f1:.4f}@th={th:.3f}")

        # save best
        if f1 > best_f1:
            best_f1 = f1; bad = 0
            ckpt_path = cfg.ckpt_dir / cfg.best_name
            torch.save({
                "model": model.state_dict(),
                "cfg": vars(cfg),
                "thresh": float(th),
            }, ckpt_path)
            print(f"** Saved best to {ckpt_path}")
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStop] best F1={best_f1:.4f}")
                break

if __name__ == "__main__":
    main()

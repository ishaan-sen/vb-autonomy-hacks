#!/usr/bin/env python3
import torch, numpy as np
from torch.utils.data import DataLoader
from config import CFG
from datasets.pickle_dataset import load_all_pickles, make_splits, AiderPickleDataset, build_transforms
from models.classifier import build_model
from utils.metrics import macro_report
from utils.common import EMA

@torch.no_grad()
def main():
    cfg = CFG(); device = "cuda" if torch.cuda.is_available() else "cpu"
    arr_by_class = load_all_pickles(cfg.DATA_DIR, cfg.classes)
    _, _, test_s  = make_splits(arr_by_class, cfg.classes, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    tf_train, tf_eval = build_transforms(cfg.image_size)
    test_loader = DataLoader(AiderPickleDataset(test_s, tf_eval), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(cfg.model_name, len(cfg.classes)).to(device)
    ckpt = torch.load(cfg.out_dir()/cfg.ckpt_best, map_location=device)
    model.load_state_dict(ckpt["model"])

    eval_model = model
    if ckpt.get("ema"):
        ema_obj = EMA(model); ema_obj.load_state_dict(ckpt["ema"])
        eval_model = ema_obj.ema

    ys, ps = [], []
    for x, y in test_loader:
        x = x.to(device)
        p = eval_model(x).argmax(dim=1).cpu().numpy()
        ys.append(y.numpy()); ps.append(p)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    rep, cm = macro_report(y_true, y_pred, cfg.classes)
    from pprint import pprint; pprint(rep)
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()

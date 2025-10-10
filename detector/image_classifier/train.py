#!/usr/bin/env python3
import os, json, math, random, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from tqdm import tqdm
from config import CFG
from datasets.pickle_dataset import load_all_pickles, make_splits, AiderPickleDataset, build_transforms
from torch.amp import autocast
from models.classifier import build_model
from utils.metrics import macro_report
from utils.common import EMA

def _eval_with_bias(model, loader, device, bias: torch.Tensor):
    model.eval()
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if bias is not None:
            logits = logits + bias  # broadcasting: (B,C) + (C,)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets.numpy().tolist())
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return macro_f1

def tune_logit_bias(model, val_loader, device, num_classes: int, classes: list[str], focus_class: str | None = "traffic_incident"):

    model.eval()
    bias = torch.zeros(num_classes, device=device)

    if focus_class is not None and focus_class in classes:
        c = classes.index(focus_class)
        search_space = np.linspace(-1.5, 0.0, 16) 
        best_f1, best_delta = -1.0, 0.0
        for delta in search_space:
            trial_bias = bias.clone()
            trial_bias[c] = float(delta)
            f1 = _eval_with_bias(model, val_loader, device, trial_bias)
            if f1 > best_f1:
                best_f1, best_delta = f1, float(delta)
        bias[c] = best_delta
        print(f"[Calib] tuned bias for class='{focus_class}': {best_delta:.4f}  (val macroF1={best_f1:.4f})")
    else:
        print("[Calib] skip bias tuning (focus_class not provided or not found)")

    return bias.detach().cpu().numpy().astype(np.float32) 

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.reduction = reduction
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        loss = ((1-pt) ** self.gamma) * ce
        if self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum": return loss.sum()
        else: return loss

def compute_class_weights(train_samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in train_samples:
        counts[y] += 1
    inv = 1.0 / np.clip(counts, 1, None)
    w = inv / inv.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)

def maybe_mixup_cutmix(x, y, alpha_mix=0.2, alpha_cut=1.0):
    B = x.size(0)
    if alpha_mix <= 0 and alpha_cut <= 0:
        return x, y, 1.0, y
    lam = 1.0
    use_cutmix = (alpha_cut > 0) and (np.random.rand() < 0.5)
    perm = torch.randperm(B, device=x.device)
    y2 = y[perm]
    if use_cutmix:
        lam = np.random.beta(alpha_cut, alpha_cut)
        H, W = x.shape[-2:]
        cut_w = int(W * math.sqrt(1-lam))
        cut_h = int(H * math.sqrt(1-lam))
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1 = np.clip(cx - cut_w//2, 0, W); x2 = np.clip(cx + cut_w//2, 0, W)
        y1 = np.clip(cy - cut_h//2, 0, H); y2c = np.clip(cy + cut_h//2, 0, H)
        x[:, :, y1:y2c, x1:x2] = x[perm, :, y1:y2c, x1:x2]
        lam = 1 - ((x2-x1)*(y2c-y1)) / (W*H + 1e-6)
    else:
        if alpha_mix > 0:
            lam = np.random.beta(alpha_mix, alpha_mix)
            x = lam * x + (1-lam) * x[perm]
    return x, y, lam, y2

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, ema_obj=None,
                    mixup_a=0.0, cutmix_a=0.0, grad_clip=1.0, amp_dtype=torch.bfloat16):
    model.train()
    tot, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", ncols=100):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, y1, lam, y2 = maybe_mixup_cutmix(x, y, alpha_mix=mixup_a, alpha_cut=cutmix_a)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', dtype=amp_dtype):
            logits = model(x)
            if isinstance(criterion, nn.CrossEntropyLoss) and (lam == 1.0):
                loss = criterion(logits, y1)
            else:
                loss = lam*criterion(logits, y1) + (1-lam)*criterion(logits, y2)

        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema_obj: ema_obj.update(model)

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            n += y.size(0)
            tot += loss.item() * y.size(0)
    return tot/n, correct/n

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=torch.bfloat16):
    model.eval()
    ys, ps = [], []
    for x, y in tqdm(loader, desc="eval", ncols=100):
        x = x.to(device, non_blocking=True)
        with autocast('cuda', dtype=amp_dtype):
            logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.append(y.numpy()); ps.append(pred)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    return y_true, y_pred

class WarmupCosine:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
        self.opt = optimizer
        self.warm = warmup_epochs
        self.total = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.epoch = 0
        self._set_lr(0.0)

    def _set_lr(self, lr):
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warm:
            lr = self.base_lr * self.epoch / max(1, self.warm)
        else:
            t = self.epoch - self.warm
            T = max(1, self.total - self.warm)
            lr = self.min_lr + 0.5*(self.base_lr - self.min_lr)*(1 + math.cos(math.pi * t / T))
        self._set_lr(lr)

def main():
    cfg = CFG(); set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    amp_dtype = torch.bfloat16 if cfg.amp_dtype.lower() == "bf16" and torch.cuda.is_bf16_supported() else torch.float16

    # data
    arr_by_class = load_all_pickles(cfg.DATA_DIR, cfg.classes)
    train_s, val_s, test_s = make_splits(arr_by_class, cfg.classes, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    tf_train, tf_eval = build_transforms(cfg.image_size)
    train_ds = AiderPickleDataset(train_s, tf_train)
    val_ds   = AiderPickleDataset(val_s, tf_eval)
    test_ds  = AiderPickleDataset(test_s, tf_eval)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    # model
    model = build_model(cfg.model_name, num_classes=len(cfg.classes)).to(device)

    # class weights / losses
    w = compute_class_weights(train_s, len(cfg.classes)).to(device)
    if cfg.use_focal:
        criterion = FocalLoss(gamma=2.0, weight=w, reduction="mean")
    else:
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema_obj = EMA(model, decay=0.999) if cfg.ema else None

    # scheduler
    sched = None
    if cfg.use_scheduler:
        sched = WarmupCosine(optimizer,
                             warmup_epochs=cfg.warmup_epochs,
                             total_epochs=cfg.epochs,
                             base_lr=cfg.lr,
                             min_lr=cfg.min_lr)

    best_macro_f1 = -1.0
    best_epoch = 0
    no_improve = 0

    for ep in range(1, cfg.epochs+1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            ema_obj=ema_obj, mixup_a=cfg.mixup_alpha, cutmix_a=cfg.cutmix_alpha,
            grad_clip=cfg.grad_clip_norm, amp_dtype=amp_dtype
        )

        # eval with EMA weights if enabled
        eval_model = ema_obj.ema if ema_obj else model
        y_true, y_pred = evaluate(eval_model, val_loader, device, amp_dtype=amp_dtype)
        rep, cm = macro_report(y_true, y_pred, cfg.classes)
        macro_f1 = rep["macro avg"]["f1-score"]

        print(f"\n[Epoch {ep}] train_loss={train_loss:.4f} acc={train_acc:.4f}  val_macroF1={macro_f1:.4f}")
        print("Per-class F1:", {c: f'{rep[c]["f1-score"]:.3f}' for c in cfg.classes})

        if sched:
            sched.step()

        torch.save(
            {"model": model.state_dict(),
             "ema": ema_obj.state_dict() if ema_obj else None,
             "cfg": cfg.__dict__},
            cfg.out_dir()/cfg.ckpt_last
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = ep
            no_improve = 0
            torch.save(
                {"model": model.state_dict(),
                 "ema": ema_obj.state_dict() if ema_obj else None,
                 "cfg": cfg.__dict__},
                cfg.out_dir()/cfg.ckpt_best
            )
            print(f"** Saved best to {cfg.out_dir()/cfg.ckpt_best} **")
        else:
            no_improve += 1

        if cfg.early_stop and no_improve >= cfg.early_stop_patience:
            print(f"[EarlyStop] No improvement for {cfg.early_stop_patience} epochs, stop at epoch {ep}. Best @ {best_epoch} (macroF1={best_macro_f1:.4f})")
            break

    # test with best
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(cfg.out_dir()/cfg.ckpt_best, map_location=device, weights_only=True)

    model.load_state_dict(ckpt["model"])
    if ckpt.get("ema"):
        ema_eval = EMA(model); ema_eval.load_state_dict(ckpt["ema"])
        eval_model = ema_eval.ema
    else:
        eval_model = model

    y_true, y_pred = evaluate(eval_model, test_loader, device, amp_dtype=amp_dtype)
    rep, cm = macro_report(y_true, y_pred, cfg.classes)
    print("\n=== TEST REPORT ===")
    from pprint import pprint; pprint(rep)
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()

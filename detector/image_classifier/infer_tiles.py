#!/usr/bin/env python3
import math, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from config import CFG
from datasets.pickle_dataset import build_transforms
from models.classifier import build_model
from utils.common import EMA

@torch.no_grad()
def load_model(device):
    cfg = CFG()
    ckpt = torch.load(Path(cfg.OUT_DIR)/cfg.ckpt_best, map_classes=None, weights_only=True, map_location=device)
    model = build_model(cfg.model_name, num_classes=len(cfg.classes)).to(device)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("ema"):
        ema = EMA(model); ema.load_state_dict(ckpt["ema"])
        model = ema.ema
    model.eval()
    return model, cfg

def tile_image(img: Image.Image, tile_size=256, stride=256):
    w, h = img.size
    tiles, boxes = [], []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tiles.append(img.crop((x, y, x+tile_size, y+tile_size)))
            boxes.append((x, y, x+tile_size, y+tile_size))
    return tiles, boxes, (w, h)

def infer_image(path, tile=256, stride=256, temperature=1.0, device="cuda"):
    model, cfg = load_model(device)
    _, tf_eval = build_transforms(cfg.image_size)
    img = Image.open(path).convert("RGB")
    tiles, boxes, (W, H) = tile_image(img, tile, stride)

    batch = torch.stack([tf_eval(t) for t in tiles]).to(device)
    logits = []
    for i in range(0, len(batch), 512):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            l = model(batch[i:i+512])
        logits.append(l.float())
    logits = torch.cat(logits, dim=0)
    probs = F.softmax(logits/temperature, dim=1).cpu().numpy()
    preds = probs.argmax(1)

    return preds, probs, boxes, cfg.classes

if __name__ == "__main__":
    import sys
    preds, probs, boxes, classes = infer_image(sys.argv[1])
    print("classes:", classes)
    print("pred counts:", np.bincount(preds, minlength=len(classes)))

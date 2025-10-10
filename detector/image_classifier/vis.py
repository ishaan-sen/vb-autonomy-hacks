import os
import sys
import pickle
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm

BASE_DIR = "/home/UFAD/zhou.zhuoyang/hacks/AIDER/data/AIDER"

# class folders exactly as shown
CLASSES = ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"]

# where to save the 5 pickle files (defaults to BASE_DIR)
OUT_DIR = BASE_DIR

# image size
SIZE = (256, 256)

# allowed extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(folder: Path):
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    # deterministic order
    files.sort()
    return files

def load_and_resize(paths):
    imgs = []
    ok_paths = []
    bad = 0
    for p in tqdm(paths, desc="Processing", unit="img"):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im = im.resize(SIZE, Image.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8)  # (H, W, 3), uint8
                imgs.append(arr)
                ok_paths.append(str(p))
        except (UnidentifiedImageError, OSError, ValueError) as e:
            bad += 1
            # optionally: print(f"[WARN] Skip {p}: {e}")
            continue
    if len(imgs) == 0:
        return np.empty((0, SIZE[1], SIZE[0], 3), dtype=np.uint8), ok_paths, bad
    imgs_np = np.stack(imgs, axis=0)  # (N, 256, 256, 3)
    return imgs_np, ok_paths, bad

def save_pickle(obj, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    base = Path(BASE_DIR)
    outdir = Path(OUT_DIR)

    if not base.exists():
        print(f"[ERROR] Base directory not found: {base}")
        sys.exit(1)

    total_imgs = 0
    for cls in CLASSES:
        cls_dir = base / cls
        if not cls_dir.exists():
            print(f"[WARN] Class folder missing: {cls_dir} (skip)")
            continue

        print(f"\n==> Class: {cls}")
        paths = list_images(cls_dir)
        print(f"Found {len(paths)} files under {cls_dir}")

        imgs_np, ok_paths, bad = load_and_resize(paths)
        total_imgs += imgs_np.shape[0]
        print(f"[{cls}] usable: {imgs_np.shape[0]}, skipped: {bad}")

        payload = {
            "class_name": cls,
            "images": imgs_np,          # (N, 256, 256, 3), uint8
            "paths": ok_paths,          # list[str]
            "size": SIZE,
            "format": "HWC_uint8_RGB_256x256"
        }
        out_pkl = outdir / f"{cls}.pkl"
        save_pickle(payload, out_pkl)
        print(f"Saved -> {out_pkl.resolve()}")

    print(f"\nDone. Total images saved across classes: {total_imgs}")

if __name__ == "__main__":
    main()

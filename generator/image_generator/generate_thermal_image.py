#!/usr/bin/env python3
"""
Sample images from a folder and apply:
  - n1 images: thermal filter
  - n2 images: thermal filter + Gaussian hotspot overlay (fire simulation)

Outputs go to:
  <output>/thermal/
  <output>/fire/

Thermal filter uses OpenCV applyColorMap if available, else Matplotlib.
Gaussian hotspot is colorized with the same colormap and blended transparently.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

# Optional OpenCV (preferred)
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Fallback Matplotlib
try:
    import matplotlib.cm as cm
    HAS_MPL = True
except Exception:
    HAS_MPL = False

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def to_uint801(arr: np.ndarray) -> np.ndarray:
    """Clip float [0..1] → uint8 [0..255]."""
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def apply_colormap_u8(gray_u8: np.ndarray, cmap_name: str = "inferno") -> np.ndarray:
    """
    Map uint8 grayscale → RGB uint8 using OpenCV or Matplotlib.
    """
    if HAS_CV2:
        cmap_map = {
            "jet": cv2.COLORMAP_JET,
            "inferno": cv2.COLORMAP_INFERNO,
            "plasma": cv2.COLORMAP_PLASMA,
            "turbo": cv2.COLORMAP_TURBO,
            "hot": cv2.COLORMAP_HOT,
            "magma": cv2.COLORMAP_MAGMA,
        }
        cv_cmap = cmap_map.get(cmap_name.lower(), cv2.COLORMAP_INFERNO)
        colored_bgr = cv2.applyColorMap(gray_u8, cv_cmap)
        rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        return rgb
    else:
        if not HAS_MPL:
            raise RuntimeError("Please install either opencv-python or matplotlib.")
        cmap = cm.get_cmap(cmap_name)
        rgb = (cmap(gray_u8.astype(np.float32) / 255.0)[..., :3] * 255.0).astype(np.uint8)
        return rgb


def thermal_filter(img: Image.Image, cmap: str = "inferno", alpha_thermal: float = 1.0) -> Image.Image:
    """
    Apply a thermal colorization to the whole image, then optionally blend with original (alpha_thermal).
    alpha_thermal=1.0 → full thermal recolor; <1.0 → semi-transparent overlay.
    """
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    rng = float(np.ptp(gray))
    if rng < 1e-6:
        gray_u8 = np.zeros_like(gray, dtype=np.uint8)
    else:
        gray_u8 = np.uint8(255.0 * (gray - gray.min()) / (rng + 1e-6))

    heat_rgb = apply_colormap_u8(gray_u8, cmap_name=cmap)
    base = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    ov = heat_rgb.astype(np.float32) / 255.0

    out = (1 - alpha_thermal) * base + alpha_thermal * ov
    return Image.fromarray(to_uint801(out))


def gaussian_hotspot_overlay(size: Tuple[int, int],
                             cmap: str = "inferno",
                             alpha_fire: float = 0.6,
                             sigma_range: Tuple[float, float] = (0.06, 0.22)) -> np.ndarray:
    """
    Create a thermal-colored Gaussian hotspot overlay (RGB float [0..1]) of given size.
    - Random center
    - Random sigma in fraction of min(H,W)
    """
    h, w = size
    yy = np.linspace(0, 1, h, dtype=np.float32)
    xx = np.linspace(0, 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(xx, yy)

    cx = np.random.uniform(0.3, 0.7)
    cy = np.random.uniform(0.3, 0.7)
    sigma_frac = np.random.uniform(*sigma_range)
    sigma = sigma_frac * min(h, w)

    # Use pixel grid for consistent sigma
    X = (xx * (w - 1))
    Y = (yy * (h - 1))
    g = np.exp(-(((X - cx * (w - 1)) ** 2 + (Y - cy * (h - 1)) ** 2) / (2.0 * (sigma ** 2) + 1e-6))).astype(np.float32)

    g = (g - g.min()) / (g.max() - g.min() + 1e-6)  # normalize 0..1
    heat_rgb = apply_colormap_u8(to_uint801(g), cmap_name=cmap).astype(np.float32) / 255.0

    # Blend weight alpha_fire will be applied later against the base
    return np.clip(heat_rgb * alpha_fire, 0.0, 1.0)


def blend_overlay(base_img: Image.Image, overlay_rgb_float01: np.ndarray) -> Image.Image:
    """Blend pre-multiplied overlay (already scaled by its alpha) over base image."""
    base = np.asarray(base_img.convert("RGB"), dtype=np.float32) / 255.0
    # Compute effective alpha per pixel from overlay brightness (optional refinement):
    # Here we just add the overlay contribution (overlay already has alpha multiplied in).
    out = np.clip(base * (1.0) + overlay_rgb_float01, 0.0, 1.0)
    return Image.fromarray(to_uint801(out))


def sample_non_overlapping(images: List[Path], n1: int, n2: int, seed: int) -> Tuple[List[Path], List[Path]]:
    rnd = random.Random(seed)
    shuffled = images[:]
    rnd.shuffle(shuffled)
    if n1 + n2 > len(shuffled):
        raise SystemExit(f"Requested n1+n2={n1+n2} > available images={len(shuffled)} (no overlap).")
    g1 = shuffled[:n1]
    g2 = shuffled[n1:n1+n2]
    return g1, g2


def main():
    ap = argparse.ArgumentParser(description="Sample images and apply thermal / thermal+fire overlays.")
    ap.add_argument("--input", required=True, type=Path, help="Folder with images")
    ap.add_argument("--output", required=True, type=Path, help="Output root folder")
    ap.add_argument("--n1", type=int, required=True, help="Number of images for THERMAL only")
    ap.add_argument("--n2", type=int, required=True, help="Number of images for THERMAL + FIRE overlay")
    ap.add_argument("--cmap", default="inferno", help="Colormap: inferno, jet, turbo, plasma, hot, magma")
    ap.add_argument("--alpha-thermal", type=float, default=0.9, help="Thermal filter strength (0..1)")
    ap.add_argument("--alpha-fire", type=float, default=0.6, help="Gaussian hotspot overlay strength (0..1)")
    ap.add_argument("--sigma-min", type=float, default=0.06, help="Hotspot sigma min (fraction of min(H,W))")
    ap.add_argument("--sigma-max", type=float, default=0.22, help="Hotspot sigma max (fraction of min(H,W))")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    input_dir = args.input
    output_dir = args.output
    out_thermal = output_dir / "thermal"
    out_fire = output_dir / "fire"
    out_thermal.mkdir(parents=True, exist_ok=True)
    out_fire.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    g1, g2 = sample_non_overlapping(images, args.n1, args.n2, seed=args.seed)

    print(f"OpenCV backend: {'YES' if HAS_CV2 else 'NO (using Matplotlib)'}")
    print(f"Thermal set: {len(g1)} images | Fire set: {len(g2)} images")

    # n1: thermal only
    for p in g1:
        try:
            img = Image.open(p).convert("RGB")
            out_img = thermal_filter(img, cmap=args.cmap, alpha_thermal=args.alpha_thermal)
            out_path = out_thermal / f"{p.stem}__thermal.png"
            out_img.save(out_path)
            print(f"[THERMAL] {p.name} -> {out_path.name}")
        except Exception as e:
            print(f"[ERR] Thermal {p.name}: {e}")

    # n2: thermal + fire overlay
    for p in g2:
        try:
            img = Image.open(p).convert("RGB")
            # First: thermal filter (semi-transparent, keep details)
            therm = thermal_filter(img, cmap=args.cmap, alpha_thermal=args.alpha_thermal)
            # Then: Gaussian hotspot overlay (simulated fire)
            h, w = therm.height, therm.width
            ov = gaussian_hotspot_overlay((h, w),
                                          cmap=args.cmap,
                                          alpha_fire=args.alpha_fire,
                                          sigma_range=(args.sigma_min, args.sigma_max))
            out_img = blend_overlay(therm, ov)
            out_path = out_fire / f"{p.stem}__thermal_fire.png"
            out_img.save(out_path)
            print(f"[FIRE] {p.name} -> {out_path.name}")
        except Exception as e:
            print(f"[ERR] Fire {p.name}: {e}")

    print(f"\n✅ Done. Outputs in: {output_dir}\n - {out_thermal}\n - {out_fire}")


if __name__ == "__main__":
    main()

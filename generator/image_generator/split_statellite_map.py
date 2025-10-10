#!/usr/bin/env python3
# split_tif_pillow.py — splits to 100x100 tiles WITHOUT georeferencing

import argparse
from pathlib import Path
from PIL import Image

def split_tif_plain(input_path: Path, outdir: Path, tile: int = 100, compress: bool = True):
    outdir.mkdir(parents=True, exist_ok=True)

    im = Image.open(input_path)
    w, h = im.size

    idx = 0
    for top in range(0, h, tile):
        for left in range(0, w, tile):
            box = (left, top, min(left + tile, w), min(top + tile, h))
            tile_img = im.crop(box)
            save_kwargs = {}
            if compress:
                save_kwargs["compression"] = "tiff_lzw"
            tile_name = f"{input_path.stem}_r{top}_c{left}_idx{idx}.tif"
            tile_img.save(outdir / tile_name, **save_kwargs)
            idx += 1
    print(f"Done. Wrote {idx} tiles to: {outdir}")

def main():
    p = argparse.ArgumentParser(description="Split a TIFF into fixed-size tiles (no georeferencing).")
    p.add_argument("input", type=Path)
    p.add_argument("--tile", type=int, default=100, help="Tile size in pixels (default 100)")
    p.add_argument("--outdir", type=Path, help="Output directory (default: <stem>_tiles_plain)")
    p.add_argument("--no-compress", action="store_true", help="Disable LZW compression")
    args = p.parse_args()

    outdir = args.outdir or args.input.with_name(f"{args.input.stem}_tiles_plain")
    split_tif_plain(args.input, outdir, args.tile, compress=not args.no_compress)

if __name__ == "__main__":
    main()

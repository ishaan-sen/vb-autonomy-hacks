#!/usr/bin/env python3
"""
Batch image-to-image generation with Google GenAI (Gemini).

For every image in INPUT_DIR, generate two images using:
  1) "Add flooding to this area"
  2) "Create traffic incident in this region"

Outputs are saved to OUTPUT_DIR with descriptive filenames.

Usage:
  python batch_genai_image2image_with_key.py \
      --input /path/to/images \
      --output /path/to/output \
      --api-key YOUR_API_KEY
"""

import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
from google import genai

MODEL_NAME = "gemini-2.5-flash-image"
PROMPTS = [
    "Add flooding to this area",
    "Create traffic incident in this region",
]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def iter_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def safe_stem(text: str) -> str:
    """Create a filesystem-safe version of a filename or prompt."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip())[:60]


def result_exists(base: str, prompt_slug: str, outdir: Path) -> bool:
    """Check if any file for this image/prompt combination already exists."""
    pattern = f"{base}__{prompt_slug}"  # prefix pattern
    for f in outdir.iterdir():
        if f.is_file() and f.name.startswith(pattern):
            return True
    return False


def generate_images_for_one(client: genai.Client, image_path: Path, outdir: Path, prompts, model):
    img = Image.open(image_path)
    base = safe_stem(image_path.stem)

    for prompt in prompts:
        pslug = safe_stem(prompt.lower().replace(" ", "-"))

        # ✅ Skip if result already exists
        if result_exists(base, pslug, outdir):
            print(f"[SKIP] Existing result found for {image_path.name} ({prompt})")
            continue

        try:
            resp = client.models.generate_content(
                model=model,
                contents=[prompt, img],
                config={"response_modalities": ["IMAGE"]},  # use uppercase
            )

            saved = 0
            for ci, cand in enumerate(getattr(resp, "candidates", []) or []):
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", []) if content else []
                for pi, part in enumerate(parts, start=1):
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data and getattr(inline_data, "data", None):
                        out_file = outdir / f"{base}__{pslug}__c{ci+1}_p{pi}.png"
                        with open(out_file, "wb") as f:
                            f.write(inline_data.data)
                        saved += 1

        except Exception as e:
            print(f"[ERROR] Generation failed for '{image_path.name}' ({prompt}): {e}")
            continue

        if saved == 0:
            print(f"[WARN] No output for {image_path.name} ({prompt})")
        else:
            print(f"[OK] {image_path.name} ({prompt}) -> {saved} file(s)")


def main():
    parser = argparse.ArgumentParser(description="Batch image2image with Google GenAI.")
    parser.add_argument("--input", required=True, type=Path, help="Input folder with images")
    parser.add_argument("--output", required=True, type=Path, help="Output folder for results")
    parser.add_argument("--api-key", required=True, help="Google GenAI API key")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name (default: gemini-2.5-flash-image)")
    args = parser.parse_args()

    input_dir, output_dir = args.input, args.output
    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=args.api_key)

    files = list(iter_images(input_dir))
    if not files:
        raise SystemExit(f"No valid images found in {input_dir}")

    print(f"Found {len(files)} images. Generating (skipping existing results)...")
    for img_path in files:
        generate_images_for_one(client, img_path, output_dir, PROMPTS, args.model)

    print(f"✅ Done. Results saved in: {output_dir}")


if __name__ == "__main__":
    main()


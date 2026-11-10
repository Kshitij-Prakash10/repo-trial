#!/usr/bin/env python3
"""OCR a photo or scan of a document: page scanner -> text detection -> text recognition -> ordered text.

    python ocr_scan.py --image path.jpg --out result.json

Writes result.json (text, lines and words with boxes and confidences) and result.txt (plain text).
Options: --weights pretrained (docTR weights without fine-tuning), --no-scanner (skip the page scanner
stage), --overlay out.png (the page with the word boxes drawn). Runs on the CPU, offline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True, help="photo or scan (JPEG, PNG, WebP, ...)")
    ap.add_argument("--out", required=True, help="result JSON path; the plain text goes next to it as .txt")
    ap.add_argument("--weights", default="finetuned", choices=["finetuned", "pretrained"])
    ap.add_argument("--no-scanner", action="store_true", help="skip the page scanner stage")
    ap.add_argument("--overlay", help="also save the processed page with the word boxes drawn")
    a = ap.parse_args(argv)
    if not os.path.exists(a.image):
        print(f"error: no such file: {a.image}", file=sys.stderr)
        return 2
    from PIL import Image, ImageDraw, ImageOps

    from ocr.pipeline import OcrPipeline
    try:
        with Image.open(a.image) as f:
            image = ImageOps.exif_transpose(f)
            image.load()
    except Exception as e:
        print(f"error: cannot read {a.image} as an image: {e}", file=sys.stderr)
        return 2
    pipe = OcrPipeline.load(a.weights, use_scanner=not a.no_scanner)
    result = pipe.run(image)
    result["image"] = os.path.abspath(a.image)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    txt = os.path.splitext(a.out)[0] + ".txt"
    with open(txt, "w") as f:
        f.write(result["text"] + "\n")
    if a.overlay:
        if result["scanner"]["used"]:
            from scanner import geometry as geo
            import numpy as np
            s = min(1.0, 2400 / max(image.size))
            work = image.resize((max(1, int(image.width * s)), max(1, int(image.height * s)))) if s < 1 else image
            page = geo.warp(work, np.asarray(result["scanner"]["corners"], float) * s)
        else:
            page = image.resize((result["page"]["width"], result["page"]["height"]))
        d = ImageDraw.Draw(page)
        for w in result["words"]:
            d.rectangle(w["box"], outline=(220, 0, 0), width=2)
        page.save(a.overlay)
    sc = result["scanner"]
    print(f"{len(result['words'])} words in {len(result['lines'])} lines, {result['seconds']} s; "
          f"scanner crop {'used' if sc['used'] else 'not used'} ({sc['reason']})")
    print(f"wrote {a.out} and {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

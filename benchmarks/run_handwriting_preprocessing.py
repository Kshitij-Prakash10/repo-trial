"""Compare handwriting image preparation with fixed crops and printed reader outputs.

The input is a paired full-page benchmark JSON, or the stored legacy validation trace.
Only the handwriting reader runs again. The printed reading and script routing stay fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from PIL import Image

from benchmarks import common as C, run_handwriting as h
from ocr import handwriting as hw
from ocr.hwdata import rect_crop
from ocr.scan import keep_printed

VARIANTS = ("baseline", "lanczos", "stroke")


def prepare(crop, variant, size):
    clean = hw.prep_line(crop)
    if variant == "lanczos":
        return np.asarray(Image.fromarray(clean).resize(size, Image.Resampling.LANCZOS))
    if variant == "stroke":
        # Keep a little more contrast in faint narrow ink strokes before processor resampling.
        gray = clean[..., 0]
        accent = cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        strengthened = cv2.addWeighted(gray, 0.7, accent, 0.3, 0)
        return np.repeat(strengthened[..., None], 3, axis=2)
    return clean

def evaluate(rows, fonts, reader, variant, image_hashes):
    scored, elapsed = [], 0.0
    pages = {}
    for row in rows:
        pages.setdefault(row["page"], []).append(row)
    size = reader.processor.image_processor.size
    size = (size["width"], size["height"])
    for seed, source in pages.items():
        page = h.render_page(seed, fonts)
        if [r["truth"] for r in source] != [ln["text"] for ln in page["lines"]]:
            raise ValueError(f"The annotation trace does not match generated page {seed}")
        if seed in image_hashes:
            actual = hashlib.sha256(page["image"].tobytes()).hexdigest()
            if actual != image_hashes[seed]:
                raise ValueError(f"Generated pixels differ from the trace for page {seed}")
        crops = [rect_crop(page["image"], ln["quad"]) for ln in page["lines"]]
        t = perf_counter()
        preds = reader.read([prepare(x, variant, size) for x in crops], prep=False)
        elapsed += perf_counter() - t
        for r, crop, (text, conf) in zip(source, crops, preds, strict=True):
            keep, _ = keep_printed(r["printed"], text, crop)
            chosen = r["routed"] and not keep and conf > r["printed_conf"]
            scored.append({**r, "handwriting": text, "handwriting_conf": conf, "keep_printed": keep,
                           "shipped": text if chosen else r["printed"]})
    results = h.score(scored)
    print(variant, "hands", results["handwriting"]["line"], "selected", results["shipped"]["line"],
          "fields", results["shipped"]["fields"]["exact"], flush=True)
    decimal_lines = [r for r in scored if h.has_decimal(r["truth"])]
    decimal_exact = {
        key: C.frac(sum(h.numbers(r[key]) == h.numbers(r["truth"]) for r in decimal_lines), len(decimal_lines))
        for key in ("handwriting", "shipped")
    }
    return {"rows": scored, "scores": results,
            "decimal_sequences_exact": decimal_exact,
            "latency_ms_per_line": 1000 * elapsed / len(scored)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    trace = json.loads(args.trace.read_text())
    image_hashes = {}
    if "pages" in trace:
        rows = [row for page in trace["pages"] for row in page["annotated_readers"]]
        image_hashes = {page["seed"]: page["image_sha256"] for page in trace["pages"]}
    else:
        rows = trace["validation"]["rows"]
    if not rows:
        parser.error("The trace must have annotated reader outputs.")

    C.limit_threads(args.threads)
    torch.manual_seed(C.SEED)
    reader = hw.HandwritingReader(local_only=True)
    fonts = h.font_files(".")
    report = {"git": C.git_info("."), "threads": args.threads,
              "trace_sha256": C.file_sha256(args.trace),
              "generator_sha256": C.file_sha256(h.__file__),
              "runner_sha256": C.file_sha256(__file__),
              "versions": {p: importlib.metadata.version(p) for p in ("pillow", "torch", "transformers")},
              "model_sha256": {p.name: C.file_sha256(p) for p in sorted(Path(reader.name).glob("*.safetensors"))},
              "font_sha256": {Path(f).name: C.file_sha256(f) for f in fonts},
              "variants": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for variant in args.variant or VARIANTS:
        report["variants"][variant] = evaluate(rows, fonts, reader, variant, image_hashes)
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()

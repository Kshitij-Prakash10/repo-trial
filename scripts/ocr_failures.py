#!/usr/bin/env python3
"""Pick recognition failure examples from an evaluation run and save their crops.

    python scripts/ocr_failures.py --system finetuned

Reads runs/eval/<system>_test/<dataset>_rec.jsonl, draws a fixed random sample (seed 2026) of the
words that were read wrongly, saves the crops for the CC BY datasets (SROIE, CORD) under
docs/ocr/failures/ and prints a Markdown table. FUNSD examples are listed as text only (its licence
is research-only, so no crops are committed).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image  # noqa: E402

from ocr.common import ROOT  # noqa: E402
from ocr.data import word_crops  # noqa: E402

SEED = 2026
PICK = {"sroie": 5, "cord": 5, "funsd": 4}
OUT = os.path.join(ROOT, "docs", "ocr", "failures")


def main(system: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    print("| Dataset | Crop | Prediction | Ground truth |\n|---|---|---|---|")
    for ds, n in PICK.items():
        with open(os.path.join(ROOT, "runs", "eval", f"{system}_test", f"{ds}_rec.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        wrong = [r for r in rows if r["char_err"] > 0]
        chosen = {r["index"]: r for r in rng.sample(wrong, n)}
        for i, (crop, text, _) in enumerate(word_crops(ds, "test")):
            if i not in chosen:
                continue
            r = chosen[i]
            assert r["true"] == text, (r, text)
            cell = "(not committed)"
            if ds != "funsd":
                name = f"{ds}_{i:06d}.png"
                im = Image.fromarray(crop)
                im = im.resize((max(8, round(im.width * 48 / im.height)), 48), Image.BICUBIC)
                im.save(os.path.join(OUT, name))
                cell = f"![{name}](docs/ocr/failures/{name})"
            esc = lambda t: "`" + t.replace("`", "'").replace("|", "\\|") + "`" if t else "(empty)"
            print(f"| {ds} | {cell} | {esc(r['pred'])} | {esc(r['true'])} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", default="finetuned")
    main(ap.parse_args().system)

#!/usr/bin/env python3
"""Build the 10-worst-failures gallery for reports/final_report.md from the evaluation rows.

    python scripts/failure_gallery.py

Reads runs/eval_v2/ (written by `python -m ocr.evaluate_v2 --task all`) and writes small JPEGs and
reports/failures/failures.json. Picks: the 4 worst GNHK lines and 3 worst IAM lines (by character
errors, lines of at least 15 characters), the 2 worst end-to-end GNHK pages, and the worst orientation
miss. Deterministic: ties are broken by id.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr import hwdata  # noqa: E402
from ocr.common import ROOT  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "eval_v2")
OUT = os.path.join(ROOT, "reports", "failures")


def rows(path: str) -> list[dict]:
    with open(os.path.join(RUNS, path)) as f:
        return [json.loads(x) for x in f]


def save(img: np.ndarray, name: str, max_side: int = 900) -> str:
    im = Image.fromarray(img)
    im.thumbnail((max_side, max_side))
    im.convert("RGB").save(os.path.join(OUT, name), quality=80)
    return f"failures/{name}"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    out = []
    gnhk = {p["id"]: p for p in hwdata.gnhk_pages("test")}
    lines = [r for r in rows("gnhk/gnhk_lines_finetuned_test.jsonl") if r["char_n"] >= 15]
    for r in sorted(lines, key=lambda r: (-r["char_err"] / r["char_n"], r["page"], r["line"]))[:4]:
        ln = next(x for x in hwdata.gnhk_lines(gnhk[r["page"]]) if x["line"] == r["line"])
        out.append({"kind": "GNHK line", "id": f"{r['page']} line {r['line']}", "true": r["true"], "pred": r["pred"],
                    "cer": round(r["char_err"] / r["char_n"], 3), "confidence": r["confidence"],
                    "image": save(ln["crop"], f"gnhk_{r['page']}_{r['line']}.jpg")})
    iam = [r for r in rows("iam/iam_finetuned_test.jsonl") if r["char_n"] >= 15]
    worst = sorted(iam, key=lambda r: (-r["char_err"] / r["char_n"], r["id"]))[:3]
    want = {r["id"]: r for r in worst}
    for i, im, _ in hwdata.iam_lines("test"):
        if i in want:
            r = want[i]
            out.append({"kind": "IAM line", "id": i, "true": r["true"], "pred": r["pred"], "cer": round(r["char_err"] / r["char_n"], 3),
                        "confidence": r["confidence"], "image": save(np.asarray(im), f"iam_{i}.jpg")})
    pages = [r for r in rows("gnhk/gnhk_pages_full_test.jsonl") if r["char_n"] >= 50]
    for r in sorted(pages, key=lambda r: (-r["char_err"] / r["char_n"], r["page"]))[:2]:
        out.append({"kind": "GNHK page, end to end", "id": r["page"], "true": r["true"][:300], "pred": r["pred"][:300],
                    "cer": round(r["char_err"] / r["char_n"], 3),
                    "image": save(hwdata.load_image(gnhk[r["page"]]["path"]), f"page_{r['page']}.jpg")})
    miss = [r for r in rows("orientation/orientation_test.jsonl") if not r["final_correct"]]
    if miss:
        r = sorted(miss, key=lambda r: (-r["confidence"], r["dataset"], r["page"], r["k"]))[0]
        path = next(p["path"] for p in hwdata.gnhk_pages("test") if os.path.basename(p["path"]) == r["page"]) \
            if r["dataset"] == "gnhk" else None
        if path is None:
            from ocr.common import load_pages
            path = next(p["path"] for p in load_pages(r["dataset"], "test") if os.path.basename(p["path"]) == r["page"])
        img = np.ascontiguousarray(np.rot90(hwdata.load_image(path), r["k"]))
        out.append({"kind": "orientation miss", "id": f"{r['dataset']} {r['page']} turned {90 * r['k']} degrees CCW",
                    "true": f"k={r['k']}", "pred": f"classifier k={r['pred']} (confidence {r['confidence']}), flip margin {r['flip_margin']}",
                    "image": save(img, f"orient_{r['dataset']}_{os.path.splitext(r['page'])[0]}_{r['k']}.jpg", 600)})
    with open(os.path.join(OUT, "failures.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for x in out:
        print(x["kind"], x["id"], x.get("cer"), "|", x["true"][:60], "|", x["pred"][:60])


if __name__ == "__main__":
    main()

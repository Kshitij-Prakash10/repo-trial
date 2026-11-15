"""Tables from the validation studies (no models needed).

    python scripts/val_studies/analyse.py

Reads reports/val_studies/*.json and prints: the 180-degree check's margin sweep, deskew's CER change
by estimated angle, and page CER for each routing rule tried.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ocr.common import error_counts, rates  # noqa: E402

V = os.path.join(ROOT, "reports", "val_studies")
SETS = ("funsd", "sroie", "cord", "gnhk")


def flip_sweep() -> None:
    r = json.load(open(os.path.join(V, "orientation_flip_val.json")))
    print("180-degree check: val accuracy over all pages x 4 turns, by margin (inf = classifier alone)")
    for m in (0.0, 0.02, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, float("inf")):
        hit = {s: [] for s in SETS}
        for x in r:
            p, d = np.array(x["p"]), x["c0"] - x["c2"]
            for k in range(4):
                res = (k - int(p[k].argmax())) % 4
                if res in (0, 2) and (d if res == 0 else -d) < -m:
                    res = (res + 2) % 4
                hit[x["src"]].append(res == 0)
        allv = [v for h in hit.values() for v in h]
        print(f"  {m:<5} all {np.mean(allv):.4f}  " + "  ".join(f"{s} {np.mean(h):.4f}" for s, h in hit.items()))


def deskew_by_angle(r: list) -> None:
    print("deskew: mean change in page CER (points) by |estimated angle|, handwriting reader off")
    rows = [x for x in r if "cer_deskew" in x]
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 16)):
        cells = []
        for s in SETS:
            d = [(x["cer_deskew"]["char_err"] - x["cer_plain"]["char_err"]) / max(x["cer_plain"]["char_n"], 1)
                 for x in rows if x["ds"] == s and lo <= abs(x["angle"]) < hi]
            cells.append(f"{s} n={len(d)} {100 * np.mean(d):+.2f}" if d else f"{s} n=0")
        print(f"  {lo}-{hi} deg: " + "  ".join(cells))


def routing(r: list) -> None:
    def text(x, rule):
        out = []
        for ln in x["lines"]:
            t = ln["text"] if ln["script"] == "handwritten" and rule(ln) else (ln["printed_text"] if ln["script"] == "handwritten" else ln["text"])
            if t.strip():
                out.append(t)
        return "\n".join(out)
    rules = {"printed only": lambda l: False}
    for s in (0.5, 0.7, 0.9):
        rules[f"share>={s}"] = (lambda s: lambda l: l["handwritten_share"] >= s)(s)
    for s in (0.5, 0.7, 0.8):
        for pc in (0.6, 0.7, 0.8, 0.9):
            rules[f"share>={s} & printed conf<{pc}"] = (lambda s, pc: lambda l: l["handwritten_share"] >= s and l["printed_confidence"] < pc)(s, pc)
        rules[f"share>={s} & hw conf>printed conf"] = (lambda s: lambda l: l["handwritten_share"] >= s and l["confidence"] > l["printed_confidence"])(s)
    print(f"routing: page CER per rule ({len(rules)} rules)")
    for name, rule in rules.items():
        res = {s: rates([error_counts(text(x, rule), x["ref"], s if s != "gnhk" else None) for x in r if x["ds"] == s])["cer"] for s in SETS}
        print(f"  {name:<34}" + "  ".join(f"{s} {v:.4f}" for s, v in res.items()))


if __name__ == "__main__":
    flip_sweep()
    r = json.load(open(os.path.join(V, "deskew_routing_val.json")))
    deskew_by_angle(r)
    routing(r)

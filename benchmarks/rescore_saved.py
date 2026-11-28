"""Rescore saved predictions with the field and decimal metrics of a given checkout.

Nothing is read again: the saved per-line rows (reference "truth" and the readings "printed",
"handwriting", "shipped") of committed results are scored with the scoring code of `--root`. Run it on
two checkouts to separate score changes caused by the evaluator from changes in what was transcribed:

    python benchmarks/rescore_saved.py --root <old checkout> --out /tmp/rescore-before.json
    python benchmarks/rescore_saved.py --root <new checkout> --out /tmp/rescore-after.json

The new checkout reports both the current metric ("fields") and the earlier one ("fields_legacy"); on
identical saved predictions, the old checkout's "fields" must equal the new checkout's "fields_legacy".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

SOURCES = (  # (file, split or None for a top-level "rows")
    ("reports/review_2026-11/baseline_0c71106/handwriting.json", "validation"),
    ("reports/review_2026-11/baseline_0c71106/handwriting.json", "test"),
    ("reports/review_2026-11/after_ce56e17/handwriting.json", "validation"),
    ("reports/review_2026-11/after_ce56e17/handwriting.json", "test"),
    ("reports/review_2026-11/after_ce56e17/handwriting_beams2.json", "test"),
    ("reports/review_2026-11/after_ce56e17/handwriting_beams4.json", "test"),
    ("reports/review_2026-11/pr14_followup/printed-selector-control.json", None),
)
READINGS = ("printed", "handwriting", "shipped")


def load_scoring(root: str):
    root = os.path.abspath(root)
    for name in [m for m in sys.modules if m.split(".")[0] == "benchmarks"]:
        del sys.modules[name]
    sys.path.insert(0, root)
    from benchmarks import run_handwriting as rh  # the checkout's own scoring code
    if not os.path.abspath(rh.__file__).startswith(root + os.sep):
        raise SystemExit(f"benchmarks imported from {rh.__file__}, not from {root}")
    return rh


def rescore(root: str, data_root: str) -> dict:
    rh = load_scoring(root)
    commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"scoring_root_commit": commit, "data_root": os.path.abspath(data_root), "sources": []}
    for rel, split in SOURCES:
        path = os.path.join(data_root, rel)
        with open(path, "rb") as fh:
            raw = fh.read()
        data = json.loads(raw)
        rows = data[split]["rows"] if split else data["rows"]
        entry = {"file": rel, "split": split, "sha256": hashlib.sha256(raw).hexdigest(), "lines": len(rows),
                 "readings": {}}
        for key in READINGS:
            pairs = [(r[key], r["truth"]) for r in rows]
            res = {"fields": rh.field_scores(pairs),
                   "wrong_decimal_lines": sum(rh.wrong_decimal(p, t) for p, t in pairs)}
            if hasattr(rh, "field_scores_legacy"):
                res["fields_legacy"] = rh.field_scores_legacy(pairs)
            if hasattr(rh, "has_decimal"):
                res["decimal_reference_lines"] = sum(rh.has_decimal(t) for _, t in pairs)
            else:
                res["decimal_reference_lines"] = sum(any("." in n for n in rh.numbers(t)) for _, t in pairs)
            entry["readings"][key] = res
        out["sources"].append(entry)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="checkout whose scoring code is used")
    ap.add_argument("--data-root", default=None, help="checkout holding the saved results (default: --root)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = rescore(a.root, a.data_root or a.root)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scanner recall at MIN_CONFIDENCE 0.60, and corner bloat on SmartDoc background01, from the stored
per-photo results only (no photos needed; they are not in the repository).

    python reports/scanner_recall/study.py > reports/scanner_recall/study.txt

Inputs: models/finetune/eval/ft_real_only__doccorner_{val,test}.csv.gz (the shipped model,
models/scanner_real.joblib = scanner_ft_real_only.joblib) and the SmartDoc files for ft_real_only and
the synthetic-only baseline. Each row has the predicted quad ("pred", photo px), its confidence and
"found" as decided at the old threshold 0.35, and for photos with a document the IoU with the truth.

What can be tested without the photos: rules that use the confidence plus the GEOMETRY of the
predicted quad (corner angles, opposite-side ratios, corners outside the photo). Image features
(edge support along the quad, text density inside) cannot be computed. Rules are chosen on
validation; the test file is read once, for the chosen rule only.
"""
from __future__ import annotations

import csv
import gzip
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL = os.path.join(ROOT, "models", "finetune", "eval")
MIN_CONF = 0.60


def load(name: str) -> list[dict]:
    with gzip.open(os.path.join(EVAL, name), "rt") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["found"] = r["found"] == "True"
        r["negative"] = r["negative"] == "True"
        r["confidence"] = float(r["confidence"])
        r["q"] = np.array(json.loads(r["pred"]), float)
        r["iou"] = float(r["iou_img"]) if r.get("iou_img") else None
        r["iou_doc"] = float(r["iou_doc"]) if r.get("iou_doc") else None
    return rows


def geometry(q: np.ndarray) -> dict:
    ang = []
    for i in range(4):
        u, v = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
        ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    side = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    ratio = max(max(side[0], side[2]) / max(min(side[0], side[2]), 1e-6),
                max(side[1], side[3]) / max(min(side[1], side[3]), 1e-6))
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return {"max_angle_dev": float(np.max(np.abs(np.array(ang) - 90))), "side_ratio": float(ratio),
            "neg_corners": int(((q[:, 0] < -2) | (q[:, 1] < -2)).sum()), "area": area}


def summary(rows: list[dict], accept) -> dict:
    pos = [r for r in rows if not r["negative"]]
    neg = [r for r in rows if r["negative"]]
    acc_p = [r for r in pos if accept(r)]
    good = [r for r in pos if r["iou"] is not None and r["iou"] >= 0.9]
    return {"found": len(acc_p) / len(pos), "false_alarms": sum(map(accept, neg)) / max(len(neg), 1),
            "good_kept": sum(map(accept, good)) / max(len(good), 1),
            "found_iou<0.5": sum(1 for r in acc_p if r["iou"] is not None and r["iou"] < 0.5) / max(len(acc_p), 1),
            "n_pos": len(pos), "n_neg": len(neg)}


def fmt(s: dict) -> str:
    return (f"found {s['found']:.3f}  false alarms {s['false_alarms']:.3f}  IoU>=0.9 kept {s['good_kept']:.3f}  "
            f"share of found with IoU<0.5 {s['found_iou<0.5']:.3f}")


def threshold_rule(t):
    return lambda r: r["found"] and r["confidence"] >= t


def combined_rule(t_low, max_dev, max_ratio):
    def accept(r):
        if not r["found"]:
            return False
        if r["confidence"] >= MIN_CONF:
            return True
        g = r["g"]
        return (r["confidence"] >= t_low and g["max_angle_dev"] <= max_dev and g["side_ratio"] <= max_ratio
                and g["neg_corners"] == 0)
    return accept


def recall_study() -> None:
    val = load("ft_real_only__doccorner_val.csv.gz")
    for r in val:
        r["g"] = geometry(r["q"])
    print("== 1. DocCorner, shipped model (ft_real_only): threshold only ==")
    print(f"validation: {summary(val, threshold_rule(0.35))['n_pos']} photos with a document, "
          f"{summary(val, threshold_rule(0.35))['n_neg']} without")
    for t in (0.35, 0.45, 0.5, 0.55, 0.6):
        print(f"  val  t={t:<4}  " + fmt(summary(val, threshold_rule(t))))
    base = summary(val, threshold_rule(MIN_CONF))
    print("\n== 2. confidence in [t_low, 0.60) accepted when the quad's geometry is plausible (chosen on val) ==")
    print("constraint: validation false alarms at most +1 point over t=0.60; objective: most pages found")
    best = None
    for t_low in (0.35, 0.4, 0.45, 0.5, 0.55):
        for dev in (5, 10, 15, 20, 25, 30):
            for ratio in (1.1, 1.2, 1.3, 1.5, 2.0):
                s = summary(val, combined_rule(t_low, dev, ratio))
                if s["false_alarms"] <= base["false_alarms"] + 0.01 and s["good_kept"] >= base["good_kept"]:
                    key = (s["found"], -s["false_alarms"])
                    if best is None or key > best[0]:
                        best = (key, (t_low, dev, ratio), s)
    (t_low, dev, ratio), s = best[1], best[2]
    print(f"  best on val: t_low={t_low} max angle deviation={dev} deg, max opposite-side ratio={ratio}, "
          f"no corner outside the photo")
    print("  val  t=0.60 alone   " + fmt(base))
    print("  val  combined       " + fmt(s))
    # how stable is it: neighbours of the chosen setting
    print("  neighbours on val (found / false alarms):")
    for dt in (-0.05, 0.0, 0.05):
        cells = []
        for dd in (-5, 0, 5):
            n = summary(val, combined_rule(round(t_low + dt, 2), dev + dd, ratio))
            cells.append(f"t_low {t_low + dt:.2f} dev {dev + dd:>2}: {n['found']:.3f}/{n['false_alarms']:.3f}")
        print("    " + " | ".join(cells))
    test = load("ft_real_only__doccorner_test.csv.gz")
    for r in test:
        r["g"] = geometry(r["q"])
    print("\n  test, read once for the chosen rule:")
    print("  test t=0.35         " + fmt(summary(test, threshold_rule(0.35))))
    print("  test t=0.60 alone   " + fmt(summary(test, threshold_rule(MIN_CONF))))
    print("  test combined       " + fmt(summary(test, combined_rule(t_low, dev, ratio))))


def bloat_study() -> None:
    print("\n== 3. SmartDoc background01: predicted area of the shipped model vs the synthetic baseline ==")
    print("The true quads are not stored, so the baseline's quad is used as the reference on frames where")
    print("the baseline itself is very accurate (its iou_doc >= 0.95; baseline Jaccard on background01 is 0.905).")
    ft = {r["id"]: r for r in load("ft_real_only__smartdoc.csv.gz")}
    bs = {r["id"]: r for r in load("base__smartdoc.csv.gz")}
    for bg in ("background01", "background02", "background03", "background04", "background05"):
        ids = [i for i in bs if i.startswith(bg + "/") and i in ft and bs[i]["iou_doc"] is not None
               and bs[i]["iou_doc"] >= 0.95 and ft[i]["iou_doc"] is not None]
        if not ids:
            continue
        ratio = np.array([geometry(ft[i]["q"])["area"] / geometry(bs[i]["q"])["area"] for i in ids])
        iou = np.array([ft[i]["iou_doc"] for i in ids])
        conf = np.array([ft[i]["confidence"] for i in ids])
        big = ratio > 1.1
        print(f"  {bg}: {len(ids)} frames; area ratio median {np.median(ratio):.3f}, >1.1 in {big.mean():.1%}; "
              f"ft iou_doc mean {iou.mean():.3f} (bloated {iou[big].mean() if big.any() else float('nan'):.3f}); "
              f"confidence of bloated frames median {np.median(conf[big]) if big.any() else float('nan'):.3f} "
              f"vs others {np.median(conf[~big]) if (~big).any() else float('nan'):.3f}")


if __name__ == "__main__":
    recall_study()
    bloat_study()

"""Draw the lowest-IoU SmartDoc Ch1 frames of an evaluated model: true quad green, prediction red.

Reads models/finetune/eval/<name>__smartdoc.csv.gz (written by scanner.finetune.report) and writes a
contact sheet plus a CSV of the cases to docs/finetune/failures/. SmartDoc 2015 Ch1 is CC BY 4.0, so
small thumbnails are committed with attribution (see docs/finetune/results.md).
CLI: python -m scanner.finetune.failures --data-home DIR --name ft_full [--n 30]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from scanner.infer import ROOT
from scanner.finetune.prep_real import MANIFEST_DIR
from scanner.finetune.report import EVAL_DIR

OUT = os.path.join(ROOT, "docs", "finetune", "failures")
CORNERS = ("tl", "tr", "br", "bl")


def sheet(data_home: str, name: str, n: int = 30, cols: int = 5, tile: int = 384) -> pd.DataFrame:
    """Lowest-n frames by official IoU, drawn on one sheet; returns the table of those frames."""
    df = pd.read_csv(os.path.join(EVAL_DIR, f"{name}__smartdoc.csv.gz"))
    man = pd.read_csv(os.path.join(MANIFEST_DIR, "smartdoc_ch1_test_frames.csv")).set_index("image_path")
    worst = df.sort_values(["iou_doc", "id"]).head(n).reset_index(drop=True)
    fr = os.path.join(data_home, "smartdoc_ch1_home", "frames")
    th = tile * 1080 // 1920
    img = Image.new("RGB", (tile * cols, (th + 34) * ((n + cols - 1) // cols)), "white")
    rows = []
    for i, r in worst.iterrows():
        g = man.loc[r["id"]]
        true = [(float(g[f"{c}_x"]), float(g[f"{c}_y"])) for c in CORNERS]
        pred = [tuple(p) for p in json.loads(r["pred"])]
        with Image.open(os.path.join(fr, r["id"])) as f:
            im = f.convert("RGB")
        d = ImageDraw.Draw(im)
        d.line(true + [true[0]], fill=(0, 200, 0), width=8)
        d.line(pred + [pred[0]], fill=(230, 0, 0), width=8)
        im = im.resize((tile, th))
        x, y = (i % cols) * tile, (i // cols) * (th + 34)
        img.paste(im, (x, y))
        ImageDraw.Draw(img).text((x + 4, y + th + 2), f"{i + 1}. IoU {r['iou_doc']:.2f}  {g['bg_name']}", fill=0)
        ImageDraw.Draw(img).text((x + 4, y + th + 16), r["id"][:52], fill=(90, 90, 90))
        rows.append({"rank": i + 1, "image_path": r["id"], "bg_name": g["bg_name"],
                     "modeltype_name": g["modeltype_name"], "iou_doc": round(float(r["iou_doc"]), 4),
                     "err_px": round(float(r["err_px"]), 1), "confidence": round(float(r["confidence"]), 3),
                     "found": bool(r["found"]),
                     "area_ratio_pred_true": round(_area(np.array(pred)) / _area(np.array(true)), 3)})
    os.makedirs(OUT, exist_ok=True)
    img.save(os.path.join(OUT, f"{name}_smartdoc_worst{n}.jpg"), quality=85)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, f"{name}_smartdoc_worst{n}.csv"), index=False)
    return out


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-home", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    print(sheet(a.data_home, a.name, a.n).to_string(index=False))

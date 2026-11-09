"""Prepare the real-photo data for fine-tuning and testing.

Inputs (downloaded during setup, never committed; see data/real/README.md):
  <data_home>/doccorner/{train,val,test}/*.parquet       DocCornerDataset (Hugging Face, pinned revision)
  <data_home>/smartdoc_ch1_home/frames/                  SmartDoc 2015 Ch1 frames.tar.gz, extracted
  <data_home>/manifests/doccorner_vs_smartdoc_dhash.csv.gz  near-duplicate check (scanner.finetune.leakcheck)

Outputs:
  <data_home>/prepared/doccorner/{train,val,test}/*.jpg + labels.csv   seeded subsets, long side <= 1024 px
  data/real/manifests/*.csv                                            committed: what was used and excluded

Corner order everywhere: TL, TR, BR, BL in image pixels of the stored image.
CLI: python -m scanner.finetune.prep_real --data-home /path/to/data
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image, ImageDraw

from scanner.infer import ROOT

SEED = 2026
MAX_SIDE = 1024  # px, stored long side for DocCorner images (the model works on a 128-cell grid)
N_SELECT = {"train": 6000, "val": 1000, "test": 2000}  # images per split (seeded, stratified by source)
LEAK_HAMMING = 12  # dHash (256 bit) distance at or below which an image counts as a SmartDoc near-duplicate
SMARTDOC_EVERY = 10  # every 10th frame of each SmartDoc video (2,489 of 24,889 frames)
MANIFEST_DIR = os.path.join(ROOT, "data", "real", "manifests")
CORNERS = ("tl", "tr", "br", "bl")


def source_of(filename: str) -> str:
    """Source prefix of a DocCornerDataset filename, e.g. 'midv500', 'negative', 'smartdoc'."""
    head = filename.lower().lstrip("_").split("_")[0].split("-")[0]
    return head or "unnamed"


def _meta(data_home: str, split: str) -> pd.DataFrame:
    """filename, is_negative and normalised corners of one split, without decoding images."""
    import pyarrow.parquet as pq
    cols = ["filename", "is_negative"] + [f"corner_{c}_{a}" for c in CORNERS for a in "xy"]
    files = sorted(os.listdir(os.path.join(data_home, "doccorner", split)))
    return pd.concat([pq.read_table(os.path.join(data_home, "doccorner", split, f), columns=cols).to_pandas()
                      for f in files if f.endswith(".parquet")], ignore_index=True)


def select(data_home: str) -> dict[str, pd.DataFrame]:
    """Seeded subsets per split after removing SmartDoc-named images and SmartDoc near-duplicates."""
    leak = pd.read_csv(os.path.join(data_home, "manifests", "doccorner_vs_smartdoc_dhash.csv.gz"))
    near = set(leak.filename[leak.hamming <= LEAK_HAMMING])
    out, excluded = {}, []
    for split, n in N_SELECT.items():
        m = _meta(data_home, split)
        m["source"] = m.filename.map(source_of)
        bad_name = m.source.eq("smartdoc")
        bad_near = m.filename.isin(near) & ~bad_name
        for reason, mask in (("named_smartdoc", bad_name), ("near_duplicate_of_smartdoc", bad_near)):
            excluded.append(m.loc[mask, ["filename", "source"]].assign(split=split, reason=reason))
        ok = m[~bad_name & ~bad_near]
        # stratified by source: each source keeps its share of the split
        frac = min(1.0, n / len(ok))
        pick = ok.groupby("source").sample(frac=frac, random_state=SEED)
        out[split] = pick.sample(frac=1.0, random_state=SEED).head(n).reset_index(drop=True)
        out[split + "_available"] = ok
    out["excluded"] = pd.concat(excluded, ignore_index=True)
    return out


def write_doccorner(data_home: str, split: str, chosen: pd.DataFrame) -> pd.DataFrame:
    """Decode the chosen images (streaming over the local Parquet files), downscale, save, return labels."""
    want = set(chosen.filename)
    meta = chosen.set_index("filename")
    dst = os.path.join(data_home, "prepared", "doccorner", split)
    os.makedirs(dst, exist_ok=True)
    ds = load_dataset("parquet", data_files=os.path.join(data_home, "doccorner", split, "*.parquet"),
                      split="train", streaming=True)
    rows = []
    for r in ds:
        fn = r["filename"]
        if fn not in want:
            continue
        im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        W0, H0 = im.size
        s = min(1.0, MAX_SIDE / max(W0, H0))
        if s < 1.0:
            im = im.resize((round(W0 * s), round(H0 * s)), Image.LANCZOS)
        W, H = im.size
        out = os.path.join(dst, fn)
        im.save(out, quality=92)
        row = {"filename": fn, "source": meta.at[fn, "source"], "is_negative": bool(r["is_negative"]),
               "orig_width": W0, "orig_height": H0, "width": W, "height": H,
               "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
        for c in CORNERS:
            x, y = r[f"corner_{c}_x"], r[f"corner_{c}_y"]
            row[f"{c}_x"] = None if x is None else float(x) * W
            row[f"{c}_y"] = None if y is None else float(y) * H
        rows.append(row)
    lab = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    lab.to_csv(os.path.join(dst, "labels.csv"), index=False)
    return lab


def smartdoc_labels(data_home: str) -> pd.DataFrame:
    """SmartDoc Ch1 test frames (every SMARTDOC_EVERY-th frame per video), corners in TL, TR, BR, BL order.
    The metadata lists tl, bl, br, tr and orders corners by the document, not by the image."""
    m = pd.read_csv(os.path.join(data_home, "smartdoc_ch1_home", "frames", "metadata.csv.gz"))
    m = m[(m.frame_index - 1) % SMARTDOC_EVERY == 0].reset_index(drop=True)
    out = m[["image_path", "bg_name", "model_name", "modeltype_name", "frame_index"]].copy()
    for c in CORNERS:
        out[f"{c}_x"], out[f"{c}_y"] = m[f"{c}_x"], m[f"{c}_y"]
    out["width"], out["height"] = 1920, 1080
    return out


def draw_check(items: list[tuple[Image.Image, np.ndarray | None, str]], path: str, cols: int = 5) -> None:
    """Contact sheet: each image with its quad (green), corner dots TL red, TR blue, BR yellow, BL magenta."""
    tiles = []
    for im, q, name in items:
        im = im.copy()
        d = ImageDraw.Draw(im)
        if q is not None:
            pts = [tuple(p) for p in q]
            w = max(2, im.width // 150)
            d.line(pts + [pts[0]], fill=(0, 220, 0), width=w)
            for (x, y), col in zip(pts, [(255, 0, 0), (0, 80, 255), (255, 220, 0), (255, 0, 255)]):
                r = max(5, im.width // 60)
                d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        im.thumbnail((360, 360))
        t = Image.new("RGB", (360, 380), "white")
        t.paste(im, (0, 0))
        ImageDraw.Draw(t).text((4, 364), name[:56], fill=(0, 0, 0))
        tiles.append(t)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (360 * cols, 380 * rows), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 360, (i // cols) * 380))
    sheet.save(path)


def quad(row) -> np.ndarray | None:
    """(4, 2) corners TL, TR, BR, BL from a labels row, or None for a negative."""
    if pd.isna(row["tl_x"]):
        return None
    return np.array([[row[f"{c}_x"], row[f"{c}_y"]] for c in CORNERS], dtype=float)


def main(data_home: str, check_dir: str) -> None:
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)
    sel = select(data_home)
    sel["excluded"].to_csv(os.path.join(MANIFEST_DIR, "doccorner_excluded.csv"), index=False)
    summary = []
    for split in N_SELECT:
        lab = write_doccorner(data_home, split, sel[split])
        lab.drop(columns=[c for c in lab.columns if c.endswith(("_x", "_y"))]).to_csv(
            os.path.join(MANIFEST_DIR, f"doccorner_{split}_used.csv"), index=False)
        summary.append({"split": split, "available_after_exclusion": len(sel[split + "_available"]),
                        "used": len(lab), "negatives": int(lab.is_negative.sum())})
        rng = np.random.default_rng(SEED)
        pick = rng.choice(len(lab), size=min(20, len(lab)), replace=False)
        root = os.path.join(data_home, "prepared", "doccorner", split)
        draw_check([(Image.open(os.path.join(root, lab.filename[i])).convert("RGB"), quad(lab.iloc[i]),
                     lab.filename[i]) for i in pick], os.path.join(check_dir, f"doccorner_{split}_20.png"))
    sd = smartdoc_labels(data_home)
    sd.to_csv(os.path.join(MANIFEST_DIR, "smartdoc_ch1_test_frames.csv"), index=False)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(sd), size=20, replace=False)
    fr = os.path.join(data_home, "smartdoc_ch1_home", "frames")
    draw_check([(Image.open(os.path.join(fr, sd.image_path[i])).convert("RGB"), quad(sd.iloc[i]),
                 sd.image_path[i]) for i in pick], os.path.join(check_dir, "smartdoc_20.png"))
    summary.append({"split": "smartdoc_ch1_test", "available_after_exclusion": 24889, "used": len(sd),
                    "negatives": 0})
    pd.DataFrame(summary).to_csv(os.path.join(MANIFEST_DIR, "summary.csv"), index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(sel["excluded"].groupby(["split", "reason"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-home", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--check-dir", default=None, help="default: <data-home>/label_checks (not committed: "
                    "DocCornerDataset images may not be redistributed)")
    a = ap.parse_args()
    main(a.data_home, a.check_dir or os.path.join(a.data_home, "label_checks"))

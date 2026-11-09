"""The stored SYNTHETIC scanner dataset in data/synthetic/ (train, val, test, ood splits).

Each split is a folder of JPEG photos (quality 90, full generated resolution) and a labels.csv with the
file name, generator seed, image size, the 4 true page corners in pixels (TL, TR, BR, BL of the page
content) and a sha256 of the file. Regenerate byte-for-byte (same Pillow / libjpeg build) with:

    python -m scanner.dataset --force
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from multiprocessing import get_context

import numpy as np
from PIL import Image

from scanner import synth

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "synthetic")
GENERATOR_VERSION = "synth-1.0"
JPEG_QUALITY = 90
# split -> (generator seeds, generator style); seed ranges are disjoint, so no test photo is seen in training
SPLITS: dict[str, tuple[range, str]] = {
    "train": (range(0, 360), "train"),
    "val": (range(100_000, 100_060), "train"),
    "test": (range(200_000, 200_200), "train"),
    "ood": (range(300_000, 300_080), "ood"),
}
CORNER_COLS = ["tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"]
COLUMNS = ["filename", "seed", "style", "width", "height", *CORNER_COLS, "kind", "background", "cut", "sha256"]


def _write_one(args: tuple[int, str, str]) -> dict:
    seed, style, folder = args
    s = synth.make_sample(seed, style)
    buf = io.BytesIO()
    s.image.save(buf, "JPEG", quality=JPEG_QUALITY)
    data = buf.getvalue()
    name = f"{seed:06d}.jpg"
    with open(os.path.join(folder, name), "wb") as f:
        f.write(data)
    row = {"filename": name, "seed": seed, "style": style, "width": s.image.size[0], "height": s.image.size[1],
           "kind": s.meta["kind"], "background": s.meta["background"], "cut": int(s.meta["cut"]),
           "sha256": hashlib.sha256(data).hexdigest()}
    row.update({c: f"{v:.3f}" for c, v in zip(CORNER_COLS, s.corners.ravel())})
    return row


def generate(data_dir: str = DATA_DIR, splits: list[str] | None = None, workers: int = 4) -> None:
    """Render every split to data_dir/<split>/ with its labels.csv (deterministic)."""
    for name in splits or list(SPLITS):
        seeds, style = SPLITS[name]
        folder = os.path.join(data_dir, name)
        os.makedirs(folder, exist_ok=True)
        with get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_write_one, [(s, style, folder) for s in seeds], chunksize=4)
        with open(os.path.join(folder, "labels.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, COLUMNS)
            w.writeheader()
            w.writerows(rows)


def exists(data_dir: str = DATA_DIR) -> bool:
    """True when every split has a labels.csv."""
    return all(os.path.exists(os.path.join(data_dir, s, "labels.csv")) for s in SPLITS)


def load_split(name: str, data_dir: str = DATA_DIR) -> list[dict]:
    """Items {"image": path, "corners": (4, 2) px TL TR BR BL, "meta": {...}} of one stored split."""
    folder = os.path.join(data_dir, name)
    items = []
    with open(os.path.join(folder, "labels.csv"), newline="") as f:
        for r in csv.DictReader(f):
            corners = np.array([float(r[c]) for c in CORNER_COLS]).reshape(4, 2)
            meta = {"seed": int(r["seed"]), "style": r["style"], "width": int(r["width"]), "height": int(r["height"]),
                    "kind": r["kind"], "background": r["background"], "cut": bool(int(r["cut"])), "split": name}
            items.append({"image": os.path.join(folder, r["filename"]), "corners": corners, "meta": meta})
    return items


def open_image(item: dict) -> Image.Image:
    """The RGB photo of an item (a path or a PIL image)."""
    im = item["image"]
    if isinstance(im, str):
        with Image.open(im) as f:
            return f.convert("RGB")
    return im.convert("RGB")


def verify(data_dir: str = DATA_DIR) -> list[str]:
    """Files whose sha256 does not match labels.csv (empty when the stored data is intact)."""
    bad = []
    for name in SPLITS:
        folder = os.path.join(data_dir, name)
        with open(os.path.join(folder, "labels.csv"), newline="") as f:
            for r in csv.DictReader(f):
                with open(os.path.join(folder, r["filename"]), "rb") as fh:
                    if hashlib.sha256(fh.read()).hexdigest() != r["sha256"]:
                        bad.append(os.path.join(name, r["filename"]))
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate or verify the synthetic scanner dataset.")
    ap.add_argument("--force", action="store_true", help="regenerate even if data/synthetic exists")
    ap.add_argument("--verify", action="store_true", help="check every file against its sha256")
    a = ap.parse_args()
    if a.verify:
        bad = verify()
        print("ok" if not bad else f"{len(bad)} files differ, e.g. {bad[:5]}")
    elif a.force or not exists():
        generate()
        print(f"written to {DATA_DIR}")
    else:
        print(f"{DATA_DIR} exists; use --force to regenerate")

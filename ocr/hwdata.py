"""Handwriting data: IAM lines (Teklia/IAM-line parquet) and GNHK pages (words with line indices).

Raw files live in data/raw/<name>/ (scripts/download_handwriting.py). Nothing in them is executed.
GNHK has no validation split, so VAL_SHARE of its training pages are set aside (seed 2026).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile

import cv2
import numpy as np
from PIL import Image

from ocr.common import DATA

RAW = os.path.join(DATA, "raw")
SEED = 2026
VAL_SHARE = 0.1
GNHK_SPECIAL = ("%math%", "%SC%", "%NA%")  # GNHK marks for formulas, scribbles and illegible words


def iam_lines(split: str, limit: int | None = None):
    """Yield (id, PIL RGB line image, text). split: train / validation / test (official RWTH split)."""
    import pyarrow.parquet as pq
    f = pq.ParquetFile(os.path.join(RAW, "iam", "data", f"{split}.parquet"))
    n = 0
    for batch in f.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            if limit is not None and n >= limit:
                return
            yield f"{split}_{n:05d}", Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"), row["text"]
            n += 1


def gnhk_extract() -> str:
    """Unpack the GNHK archive (images and JSON only, flat names) into data/raw/gnhk/pages/<split>/."""
    out = os.path.join(RAW, "gnhk", "pages")
    if os.path.isdir(out):
        return out
    with zipfile.ZipFile(os.path.join(RAW, "gnhk", "GNHK-dataset.zip")) as z:
        for part, split in (("train_data.zip", "train"), ("test_data.zip", "test")):
            os.makedirs(os.path.join(out + ".part", split), exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(z.read(part))) as zz:
                for n in zz.namelist():
                    base = os.path.basename(n)
                    if base.lower().endswith((".jpg", ".json")) and not base.startswith("."):
                        with open(os.path.join(out + ".part", split, base), "wb") as f:
                            f.write(zz.read(n))
    os.replace(out + ".part", out)
    return out


def gnhk_pages(split: str) -> list[dict]:
    """Pages of a split: {"id", "path", "words": [{"text", "poly" (4x2), "line", "type" H|P}]}.

    split: train / val / test. val is VAL_SHARE of the official training pages (fixed seed)."""
    root = gnhk_extract()
    src = "test" if split == "test" else "train"
    ids = sorted(f[:-5] for f in os.listdir(os.path.join(root, src)) if f.endswith(".json"))
    if src == "train":
        order = np.random.default_rng(SEED).permutation(len(ids))
        n_val = int(round(VAL_SHARE * len(ids)))
        pick = order[:n_val] if split == "val" else order[n_val:]
        ids = sorted(ids[i] for i in pick)
        if split == "train":  # a training page that is byte-identical to a val or test page is dropped
            held = {md5(q["path"]) for s in ("val", "test") for q in gnhk_pages(s)}
            ids = [i for i in ids if md5(os.path.join(root, src, i + ".jpg")) not in held]
    pages = []
    for i in ids:
        with open(os.path.join(root, src, i + ".json")) as f:
            words = [{"text": w["text"], "line": int(w["line_idx"]), "type": w.get("type", "H"),
                      "poly": [[w["polygon"][f"x{k}"], w["polygon"][f"y{k}"]] for k in range(4)]} for w in json.load(f)]
        pages.append({"id": i, "path": os.path.join(root, src, i + ".jpg"), "words": words})
    return pages


def load_image(path: str) -> np.ndarray:
    """RGB array with the EXIF rotation applied. GNHK's annotations refer to the rotated image:
    about a fifth of its photos are stored sideways or upside down with an EXIF orientation tag."""
    from PIL import ImageOps
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def rect_crop(img: np.ndarray, points: np.ndarray, pad: float = 0.08) -> np.ndarray:
    """Crop the minimum-area rectangle around points (N, 2) and turn it upright (text left to right)."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.asarray(points, np.float32))
    if w < h:  # make the long side horizontal
        w, h, ang = h, w, ang + 90
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    w, h = w + 2 * pad * h, h * (1 + 2 * pad)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy
    return cv2.warpAffine(img, M, (max(2, int(round(w))), max(2, int(round(h)))), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def gnhk_lines(page: dict, img: np.ndarray | None = None) -> list[dict]:
    """Text lines of a GNHK page: {"text", "type" (H, P or mixed), "special", "poly_points", "crop"}.

    Words keep the annotation order inside a line when sorted by their left edge. "special" marks
    lines with a GNHK special token; they are left out of recognition scoring."""
    if img is None:
        img = load_image(page["path"])
    lines: dict[int, list[dict]] = {}
    for w in page["words"]:
        lines.setdefault(w["line"], []).append(w)
    out = []
    for k in sorted(lines):
        ws = sorted(lines[k], key=lambda w: min(p[0] for p in w["poly"]))
        pts = np.array([p for w in ws for p in w["poly"]], np.float32)
        types = {w["type"] for w in ws}
        out.append({"line": k, "text": " ".join(w["text"] for w in ws),
                    "type": types.pop() if len(types) == 1 else "mixed",
                    "special": any(w["text"] in GNHK_SPECIAL for w in ws),
                    "points": pts, "crop": rect_crop(img, pts)})
    return out


def md5(path_or_bytes) -> str:
    b = path_or_bytes if isinstance(path_or_bytes, bytes) else open(path_or_bytes, "rb").read()
    return hashlib.md5(b).hexdigest()

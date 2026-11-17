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


LINE_MIN_ASPECT = 2.0  # several words spanning a rectangle at least this elongated are a line along it


def _text_axis(points: np.ndarray) -> tuple[tuple[float, float], float, float, float]:
    """Minimum-area rectangle around points as (centre, width, height, angle in degrees): the width
    runs along the text.

    The text direction is the rectangle's long side when that side is within 45 degrees of
    horizontal (the page is upright: EXIF, orientation classifier), or when the points come from
    several words (more than 4 points) spread at least LINE_MIN_ASPECT times longer than high.
    Otherwise (one word box taller than wide, e.g. "I" or "1") the side nearest horizontal is used,
    so a narrow word is not turned on its side."""
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    # OpenCV: the side of length w runs at `ang` degrees, the side of length h at ang + 90
    sides = [(float(w), float(h), (ang + 90.0) % 180.0 - 90.0), (float(h), float(w), (ang + 180.0) % 180.0 - 90.0)]
    long_side = max(sides, key=lambda t: t[0])
    flat_side = min(sides, key=lambda t: (round(abs(t[2]), 6), -t[0]))  # nearest horizontal; tie: longer
    several_words = len(pts) > 4 and long_side[0] >= LINE_MIN_ASPECT * max(long_side[1], 1e-6)
    width, height, angle = long_side if (abs(long_side[2]) <= 45.0 or several_words) else flat_side
    return (float(cx), float(cy)), width, height, angle


def rect_crop(img: np.ndarray, points: np.ndarray, pad: float = 0.08, words: list | None = None,
              mask_dilate: float = 0.15) -> np.ndarray:
    """Crop the minimum-area rectangle around points (N, 2) and turn it upright (text left to right).

    words (optional, off by default): the polygons ((K, 2) arrays, e.g. the 4 corners of each word box)
    of the line's words, in the same pixels as img. Then, after turning the crop upright, everything
    outside the union of those words (each grown by mask_dilate x the median word height) is painted
    with the paper colour, so that words of neighbouring lines caught by the rectangle of a curved or
    gappy line are not read. Not validated on real handwriting (GNHK), hence off by default."""
    (cx, cy), w, h, ang = _text_axis(points)
    w, h = w + 2 * pad * h, h * (1 + 2 * pad)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy
    size = (max(2, int(round(w))), max(2, int(round(h))))
    crop = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if words is not None and len(words):
        crop = mask_outside(crop, [np.asarray(q, np.float64) @ M[:, :2].T + M[:, 2] for q in words], mask_dilate)
    return crop


def mask_outside(crop: np.ndarray, polys: list, dilate: float = 0.15) -> np.ndarray:
    """Paint pixels of crop outside the (grown) union of polys with the local paper colour.

    The paper colour is the per-channel median of the brighter half of the pixels inside the kept
    area (ink is the darker part)."""
    H, W = crop.shape[:2]
    keep = np.zeros((H, W), np.uint8)
    heights = []
    for q in polys:
        q = np.asarray(q, np.float64).reshape(-1, 2)
        heights.append(float(q[:, 1].max() - q[:, 1].min()))
        cv2.fillConvexPoly(keep, np.round(cv2.convexHull(q.astype(np.float32))).astype(np.int32).reshape(-1, 2), 1)
    r = int(round(dilate * float(np.median(heights)))) if heights else 0
    if r > 0:
        keep = cv2.dilate(keep, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    keep = keep.astype(bool)
    if not keep.any() or keep.all():
        return crop
    inside = crop[keep].reshape(-1, crop.shape[2] if crop.ndim == 3 else 1)
    grey = inside.mean(1)
    paper = np.median(inside[grey >= np.median(grey)], 0)
    out = crop.copy()
    out[~keep] = paper.astype(crop.dtype).reshape(out[~keep].shape[1:]) if crop.ndim == 3 else paper.astype(crop.dtype)[0]
    return out


def gnhk_lines(page: dict, img: np.ndarray | None = None, mask_neighbours: bool = False) -> list[dict]:
    """Text lines of a GNHK page: {"text", "type" (H, P or mixed), "special", "poly_points", "crop"}.

    Words keep the annotation order inside a line when sorted by their left edge. "special" marks
    lines with a GNHK special token; they are left out of recognition scoring. mask_neighbours paints
    out everything outside the line's own words (see rect_crop; off by default, not yet validated)."""
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
                    "points": pts,
                    "crop": rect_crop(img, pts, words=[w["poly"] for w in ws] if mask_neighbours else None)})
    return out


def md5(path_or_bytes) -> str:
    b = path_or_bytes if isinstance(path_or_bytes, bytes) else open(path_or_bytes, "rb").read()
    return hashlib.md5(b).hexdigest()

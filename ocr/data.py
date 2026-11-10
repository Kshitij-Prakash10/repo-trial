"""Word-level training and evaluation data from the common annotation format.

FUNSD and CORD are annotated per word. SROIE is annotated per text line, so its lines are split into
words here with a fixed, model-free rule (`split_line`): the line crop is binarised, and the widest
runs of empty columns are taken as the spaces between the words of the transcription. Lines where the
rule cannot find enough gaps are kept out of the word-level sets and counted (see `sroie_words`).
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from ocr.common import DATA, load_pages

MIN_SIDE = 3  # px; smaller boxes are skipped as unusable crops


def _otsu(a: np.ndarray) -> int:
    hist = np.bincount(a.ravel(), minlength=256).astype(float)
    w0 = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    w1 = w0[-1] - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w0 - mu * w0[-1]) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1
    return int(np.argmax(between))


def split_line(gray: np.ndarray, box: list[int], n_words: int) -> list[list[int]] | None:
    """Word boxes (page px) for a line with n_words words, or None when the gaps cannot be found.

    gray is the whole page (uint8). Ink is the darker side of an Otsu threshold on the line crop."""
    x0, y0, x1, y1 = box
    crop = gray[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0 or crop.shape[0] < MIN_SIDE or crop.shape[1] < MIN_SIDE:
        return None
    ink = crop <= _otsu(crop)
    if ink.mean() > 0.5:  # light text on a dark background
        ink = ~ink
    cols = ink.any(axis=0)
    if not cols.any():
        return None
    first, last = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    gaps, start = [], None  # runs of empty columns strictly inside the inked span
    for x in range(first, last):
        if not cols[x]:
            start = x if start is None else start
        elif start is not None:
            gaps.append((x - start, start, x))
            start = None
    if len(gaps) < n_words - 1:
        return None
    cuts = sorted(g[1:] for g in sorted(gaps, reverse=True)[:n_words - 1])
    edges = [first] + [v for a, b in cuts for v in (a, b)] + [last]
    out = []
    for a, b in zip(edges[0::2], edges[1::2]):
        rows = ink[:, a:b].any(axis=1)
        if not rows.any():
            return None
        top, bot = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
        out.append([x0 + a, y0 + top, x0 + b, y0 + bot])
    return out


def sroie_words(split: str) -> list[dict]:
    """SROIE pages with word-level items derived from the line boxes (cached as words_<split>.jsonl).

    Each page gains "unsplit": the number of its lines that could not be split."""
    cache = os.path.join(DATA, "sroie", f"words_{split}.jsonl")
    if not os.path.exists(cache):
        with open(cache, "w") as f:
            for p in load_pages("sroie", split):
                with Image.open(p["path"]) as im:
                    gray = np.asarray(im.convert("L"))
                items, unsplit = [], 0
                for it in p["items"]:
                    words = it["text"].split()
                    boxes = split_line(gray, it["box"], len(words))
                    if boxes is None:
                        unsplit += 1
                        continue
                    items += [{"text": w, "box": b, "quad": None, "line": it["line"]} for w, b in zip(words, boxes)]
                q = {k: v for k, v in p.items() if k != "path"}
                f.write(json.dumps(dict(q, level="word", items=items, unsplit=unsplit, lines=len(p["items"])),
                                   ensure_ascii=False) + "\n")
    with open(cache) as f:
        pages = [json.loads(line) for line in f]
    for p in pages:
        p["path"] = os.path.join(DATA, "sroie", p["image"])
    return pages


def word_pages(dataset: str, split: str) -> list[dict]:
    """Pages with word-level items for any dataset."""
    return sroie_words(split) if dataset == "sroie" else load_pages(dataset, split)


def usable(it: dict) -> bool:
    x0, y0, x1, y1 = it["box"]
    return x1 - x0 >= MIN_SIDE and y1 - y0 >= MIN_SIDE and bool(it["text"].strip())


def word_crops(dataset: str, split: str, limit_pages: int | None = None):
    """Yield (crop RGB uint8 array, text, page id) for every usable word of a split."""
    for p in word_pages(dataset, split)[:limit_pages]:
        with Image.open(p["path"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        H, W = rgb.shape[:2]
        for it in p["items"]:
            if not usable(it):
                continue
            x0, y0, x1, y1 = it["box"]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 >= MIN_SIDE and y1 - y0 >= MIN_SIDE:
                yield rgb[y0:y1, x0:x1].copy(), it["text"], p["id"]

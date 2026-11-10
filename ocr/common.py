"""Shared pieces of the OCR pipeline: annotation loading, reading order, and the metrics.

Boxes are [x0, y0, x1, y1] in pixels. Text is compared after `normalise` (whitespace collapsed, case kept).
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
DATASETS = ("funsd", "sroie", "cord")
# SROIE's transcriptions are upper case even where the receipt prints mixed case ("Total" is labelled
# "TOTAL"), so text from these datasets is compared without case. FUNSD and CORD keep case.
CASE_INSENSITIVE = ("sroie",)
SEGMENT_GAP = 2.0  # a horizontal gap wider than this many line heights splits a line into segments


def load_pages(dataset: str, split: str) -> list[dict]:
    """Pages of one split in the common format written by scripts/download_data.py."""
    path = os.path.join(DATA, dataset, f"{split}.jsonl")
    with open(path) as f:
        pages = [json.loads(line) for line in f]
    for p in pages:
        p["path"] = os.path.join(DATA, dataset, p["image"])
    return pages


def normalise(text: str) -> str:
    """Collapse all whitespace to single spaces and strip; case and punctuation are kept."""
    return " ".join(text.split())


# ---------------------------------------------------------------- reading order

def group_lines(boxes: np.ndarray) -> list[list[int]]:
    """Group boxes into text lines, top to bottom, each line left to right. Returns lists of indices.

    A box joins a line when its vertical centre lies inside the line's current vertical extent
    (the median top and bottom of the boxes already in the line); otherwise it starts a new line."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if len(boxes) == 0:
        return []
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    lines: list[list[int]] = []
    ext: list[tuple[float, float]] = []
    for i in np.argsort(cy, kind="stable"):
        placed = False
        for k in range(len(lines) - 1, max(-1, len(lines) - 6), -1):  # only recent lines can match
            top, bot = ext[k]
            if top <= cy[i] <= bot and min(bot, boxes[i, 3]) - max(top, boxes[i, 1]) >= 0.4 * (boxes[i, 3] - boxes[i, 1]):
                lines[k].append(int(i))
                b = boxes[lines[k]]
                ext[k] = (float(np.median(b[:, 1])), float(np.median(b[:, 3])))
                placed = True
                break
        if not placed:
            lines.append([int(i)])
            ext.append((float(boxes[i, 1]), float(boxes[i, 3])))
    order = np.argsort([np.median(boxes[ln][:, 1] + boxes[ln][:, 3]) / 2 for ln in lines], kind="stable")
    return [sorted(lines[k], key=lambda j: boxes[j, 0]) for k in order]


def split_segments(line: list[int], boxes: np.ndarray, gap: float = SEGMENT_GAP) -> list[list[int]]:
    """Split one line (indices, left to right) where the horizontal gap exceeds gap x the line height."""
    boxes = np.asarray(boxes, float)
    h = float(np.median(boxes[line][:, 3] - boxes[line][:, 1]))
    out = [[line[0]]]
    for a, b in zip(line, line[1:]):
        if boxes[b, 0] - boxes[a, 2] > gap * h:
            out.append([])
        out[-1].append(b)
    return out


def union(boxes: np.ndarray) -> list[float]:
    b = np.asarray(boxes, float).reshape(-1, 4)
    return [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())]


def page_text(texts: list[str], boxes: np.ndarray) -> str:
    """Plain text in reading order: one text line per output line, items separated by a space."""
    return "\n".join(" ".join(texts[i] for i in ln) for ln in group_lines(boxes))


# ---------------------------------------------------------------- metrics

def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings for characters, lists for words)."""
    from rapidfuzz.distance import Levenshtein
    return int(Levenshtein.distance(a, b))


def error_counts(pred: str, true: str, dataset: str | None = None) -> dict:
    """Character and word edit counts for one pair, after normalise(); case is folded for the datasets
    in CASE_INSENSITIVE."""
    p, t = normalise(pred), normalise(true)
    if dataset in CASE_INSENSITIVE:
        p, t = p.upper(), t.upper()
    return {"char_err": edit_distance(p, t), "char_n": len(t),
            "word_err": edit_distance(p.split(), t.split()), "word_n": len(t.split())}


def rates(counts: list[dict]) -> dict:
    """Corpus-level CER and WER: total edits over total reference length (not a mean of per-item rates)."""
    ce, cn = sum(c["char_err"] for c in counts), sum(c["char_n"] for c in counts)
    we, wn = sum(c["word_err"] for c in counts), sum(c["word_n"] for c in counts)
    return {"cer": round(ce / cn, 4) if cn else None, "wer": round(we / wn, 4) if wn else None,
            "chars": cn, "words": wn, "n": len(counts)}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (N, 4) with every box in b (M, 4)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: np.clip(r[:, 2] - r[:, 0], 0, None) * np.clip(r[:, 3] - r[:, 1], 0, None)
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def match_boxes(pred: np.ndarray, true: np.ndarray, thr: float = 0.5) -> list[tuple[int, int]]:
    """One-to-one matches (pred index, true index) with IoU >= thr, best IoU first."""
    m = iou_matrix(pred, true)
    pairs = []
    if m.size:
        used_p, used_t = set(), set()
        for flat in np.argsort(-m, axis=None):
            i, j = divmod(int(flat), m.shape[1])
            if m[i, j] < thr:
                break
            if i not in used_p and j not in used_t:
                used_p.add(i)
                used_t.add(j)
                pairs.append((i, j))
    return pairs


def prf(tp: int, n_pred: int, n_true: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "tp": tp, "n_pred": n_pred, "n_true": n_true}

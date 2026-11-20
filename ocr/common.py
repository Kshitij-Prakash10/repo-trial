"""Shared pieces of the OCR pipeline: annotation loading, reading order, and the metrics.

Boxes are [x0, y0, x1, y1] in pixels. Text is compared after `normalise` (whitespace collapsed, case kept).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
DATASETS = ("funsd", "sroie", "cord")
# SROIE's transcriptions are upper case even where the receipt prints mixed case ("Total" is labelled
# "TOTAL"), so text from these datasets is compared without case. FUNSD and CORD keep case.
CASE_INSENSITIVE = ("sroie",)
SEGMENT_GAP = 2.0  # a horizontal gap wider than this many line heights splits a line into segments
# Fine-tuned weights (ocr/models.py loads them; ocr/api.py checks for them without importing torch).
MODEL_DIR = os.path.join(ROOT, "models", "ocr")
DET_CKPT = os.path.join(MODEL_DIR, "det_fast_base_ft.pt")
REC_CKPT = os.path.join(MODEL_DIR, "rec_crnn_mobilenet_v3_large_ft.pt")


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

def group_lines(boxes: np.ndarray, split_gap: float | None = None) -> list[list[int]]:
    """Group boxes into text lines, top to bottom, each line left to right. Returns lists of indices.

    A box joins a line when its vertical centre lies inside the line's current vertical extent
    (the median top and bottom of the boxes already in the line) and it overlaps that extent by at
    least 0.4 of its own height; otherwise it starts a new line. Every earlier line is a candidate
    (the most recent match wins); lines whose bottom lies above the box's top cannot match.

    split_gap (off by default): also split each line where the horizontal gap between neighbouring
    boxes exceeds split_gap x the line height (see split_segments), so side-by-side columns become
    separate lines. The segments of one line stay next to each other, left to right. It changes the
    reading order of forms and receipts and has not been validated on FUNSD / SROIE / CORD, so the
    pipeline does not use it."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if len(boxes) == 0:
        return []
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    lines: list[list[int]] = []
    tops: list[float] = []
    bots: list[float] = []
    for i in np.argsort(cy, kind="stable"):
        k = -1
        if lines:
            t, b = np.asarray(tops), np.asarray(bots)
            y0, y1 = boxes[i, 1], boxes[i, 3]
            ok = (b >= y0) & (t <= cy[i]) & (cy[i] <= b) & (np.minimum(b, y1) - np.maximum(t, y0) >= 0.4 * (y1 - y0))
            hits = np.flatnonzero(ok)
            k = int(hits[-1]) if len(hits) else -1
        if k >= 0:
            lines[k].append(int(i))
            bb = boxes[lines[k]]
            tops[k], bots[k] = float(np.median(bb[:, 1])), float(np.median(bb[:, 3]))
        else:
            lines.append([int(i)])
            tops.append(float(boxes[i, 1]))
            bots.append(float(boxes[i, 3]))
    order = np.argsort([np.median(boxes[ln][:, 1] + boxes[ln][:, 3]) / 2 for ln in lines], kind="stable")
    out = [sorted(lines[k], key=lambda j: boxes[j, 0]) for k in order]
    if split_gap is not None:
        out = [seg for ln in out for seg in split_segments(ln, boxes, split_gap)]
    return out


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


def numeric_line_accuracy(pairs) -> dict:
    """Exact transcription of lines containing a reference digit, after whitespace normalisation."""
    exact = total = 0
    for prediction, reference in pairs:
        if any(char.isdecimal() for char in reference):
            total += 1
            exact += normalise(prediction) == normalise(reference)
    return {"exact": exact, "total": total, "accuracy": round(exact / total, 4) if total else None}


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


# ---------------------------------------------------------------- numbers and units

# Dose and measurement units, lower case. Many of them are also words, initials or abbreviations
# ("Dr L Example", "BD Diagnostics", "Department of Mol Biology", "Tab Pharmacy"), so a unit counts only
# when it is written next to a number (or next to what a reader makes of one, see _NUMBER_LIKE), or when
# it is spelt in its usual case (_USUAL_SPELLING; mL and mg, but not ML or MG). The short ones that
# collide with ordinary words or initials in every case (l, g, mol, iu, unit(s)) count only next to a number.
_UNIT_WORDS = frozenset((
    "mcg", "µg", "ug", "mg", "g", "kg", "ml", "l", "dl", "mmol", "mol", "miu", "iu", "unit", "units",
    "tab", "tabs", "tablet", "tablets", "cap", "caps", "capsule", "capsules", "puff", "puffs", "drop", "drops",
    "mmhg", "bpm", "od", "bd", "tds", "qds", "prn", "nocte", "mane"))
_USUAL_SPELLING = frozenset((
    "mcg", "µg", "ug", "mg", "kg", "ml", "mL", "dl", "dL", "mmol", "mIU", "mmHg", "bpm",
    "tab", "tabs", "tablet", "tablets", "cap", "caps", "capsule", "capsules", "puff", "puffs", "drop", "drops",
    "od", "bd", "tds", "qds", "prn", "nocte", "mane"))
# A number, or a number whose digits a reader turned into look-alike letters (O/o for 0, S for 5, Z for 2,
# I/l/| for 1, B for 8: "Z.S", "SOO", "lO", "O"), or a number word.
_NUMBER_LIKE = re.compile(r"[0-9OoSZIlB|]+(?:[.,][0-9OoSZIlB|]+)*")
_NUMBER_WORDS = frozenset(("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "half"))
_EDGE_PUNCT = "()[]{}:;,.\"'"


def _parts(text: str) -> list[str]:
    """Whitespace-separated tokens without edge punctuation, compound units split at "/" (mmol/L)."""
    out = []
    for chunk in (text or "").split():
        out.extend(p for p in chunk.strip(_EDGE_PUNCT).split("/") if p)
    return out


def _number_like(part: str) -> bool:
    return bool(_NUMBER_LIKE.fullmatch(part)) or part.lower() in _NUMBER_WORDS


def has_number_or_unit(text: str) -> bool:
    """True when the text holds a digit, a percent sign or a dose / measurement unit as a word of its own.

    A unit counts when it follows a number-like token or another unit that counts ("2.5 mg", "10 units",
    "1 tab od", "InSulin O Units", "S ML") or when it is spelt in its usual case ("Omeprazole mg",
    "take two tabs nocte"). Unit-like words on their own do not count: "Dr L Example", "Mehta OD Clinic",
    "Test Result Units Reference range"."""
    text = text or ""
    if any(c.isdigit() for c in text) or "%" in text:
        return True
    prev_counts = False  # the previous token is number-like or a unit that counted
    for part in _parts(text):
        is_unit = part.isalpha() and part.lower() in _UNIT_WORDS
        if is_unit and (part in _USUAL_SPELLING or prev_counts):
            return True
        prev_counts = _number_like(part)
    return False


def number_lost(text: str) -> bool:
    """True when a unit counts (see has_number_or_unit) but the text has no digit: the number that went
    with it was probably lost or misread ("Omeprazole mg", "Insulin glargine IZ Units")."""
    return has_number_or_unit(text) and not any(c.isdigit() for c in text or "")


# Misread units that a printed reader makes of mg, mL and mcg ("Metformin SOO my").
_MISREAD_UNITS = frozenset(("my", "rng", "mq", "mI", "rnl", "rnL", "rncg", "mcq", "rnmol"))
_NOT_A_NUMBER = frozenset(("so", "is", "os", "sos", "oz"))  # words made of look-alike letters only


def has_misread_number(text: str) -> bool:
    """True when a token of look-alike letters, with no real digit or with digits mixed with look-alike
    letters, stands right before a unit or a misread unit ("Metformin SOO my", "Atorvastatin ZO mg").
    Used only when Scanner(flag_misread_digits=True): off by default, see ocr/scan.py."""
    parts = _parts(text)
    for a, b in zip(parts, parts[1:]):
        if not _NUMBER_LIKE.fullmatch(a) or not any(c.isalpha() for c in a) or a.lower() in _NOT_A_NUMBER:
            continue
        if b in _MISREAD_UNITS or (b.isalpha() and b.lower() in _UNIT_WORDS):
            return True
    return False


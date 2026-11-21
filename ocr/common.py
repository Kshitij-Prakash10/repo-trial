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


def _word_like(part: str) -> bool:
    """A capitalised word ("Bo", "So"): an initial capital, then lower-case letters with a vowel among them.
    The only lower-case look-alikes are o (0) and l (1), so a name-shaped token of look-alikes needs an "o"
    to be a word; "Il", "Sl", "Bl", "Ill" (11, 51, 81, 111) have no vowel and stay misread numbers."""
    return (len(part) > 1 and part.isalpha() and part[0].isupper() and part[1:].islower()
            and any(ch in "aeiouy" for ch in part[1:]))


def _number_like(part: str) -> bool:
    if _word_like(part):
        return False
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


# Frequency words carry no amount, so they need no quantity of their own ("Paracetamol 500 mg as needed prn").
_FREQUENCY_WORDS = frozenset(("od", "bd", "tds", "qds", "prn", "nocte", "mane"))
# A quantity: a number ("20", "2.5", "0,5"; ".5" once _parts strips the leading dot) or a range ("1-2",
# "2.5–5"). It must hold at least one real digit (see _is_quantity): look-alike letters mixed with digits
# ("5OO", which has_misread_number flags) count, as they did under the old whole-line rule, but a token of
# look-alike letters or a number word alone ("O", "SOO", "two") is not a quantity ("InSulin O Units").
_NUM = r"[0-9OoSZIlB|]+(?:[.,][0-9OoSZIlB|]+)*"
_QUANTITY = re.compile(rf"{_NUM}(?:[-–]{_NUM})?")
_QUANTITY_UNIT = re.compile(rf"({_NUM}(?:[-–]{_NUM})?)([^\W\d_]+)")  # "20mg", "2.5mL", "1-2tabs"
_LIST_INDEX = re.compile(r"\(?\d{1,3}[.):]\)?")  # "1.", "2)", "(3)", "4:" at the start of a line


def _is_quantity(part: str) -> bool:
    return bool(_QUANTITY.fullmatch(part)) and any(c.isdigit() for c in part)


# Count units (how many tablets, capsules, drops, puffs). A number word ("one tablet", "two tabs", "half a
# tab") or "a" / "an" is their quantity too; a dose or measurement unit (mg, mL, units, ...) needs digits.
_COUNT_UNITS = frozenset(("tab", "tabs", "tablet", "tablets", "cap", "caps", "capsule", "capsules",
                          "puff", "puffs", "drop", "drops"))
# The plural of a count unit with no number is the name of the form ("Paracetamol tablets as needed",
# "Eye drops nocte"), not a count whose number was lost.
_COUNT_PLURALS = frozenset(("tabs", "tablets", "caps", "capsules", "puffs", "drops"))
# ... unless the plural stands between a capitalised medicine and a frequency word: "Salbutamol puffs prn"
# lost its count ("2 puffs"). Words that name the form with the plural are not medicines ("Eye drops nocte").
_FORM_NAMES = frozenset(("eye", "ear", "nasal", "nose", "mouth", "throat", "skin"))
# Header words of a lab table: a line of them and a unit is a column header, not a value that went missing
# ("Analyte (mmol/L)", "Test Units"); a test name with a unit and no value is ("Glucose (mmol/L)").
_HEADER_WORDS = frozenset(("analyte", "test", "tests", "parameter", "parameters", "investigation",
                           "investigations", "result", "results", "value", "values", "unit", "units",
                           "reference", "range", "interval", "flag", "name", "method", "specimen"))
_COUNT_WORDS = _NUMBER_WORDS | {"a", "an"}
# Unit words that count only next to a number in has_number_or_unit (they collide with initials and words:
# "G Sample", "Department of Mol Biology"). Right before a unit or frequency word that counts they are
# units too ("Paracetamol g prn", "Insulin units bd").
_AMBIGUOUS_UNITS = frozenset(("g", "l", "mol", "iu", "unit", "units"))
_DIGITS = re.compile(r"\d+(?:[.,]\d+)*(?:[-\u2013]\d+(?:[.,]\d+)*)?")  # a value written in digits only
# Words that take the number before them, so that number is not the value of a unit written before it
# ("Omeprazole mg 7 days", "Omeprazole mg x 2 weeks").
_TAKES_NUMBER = frozenset(("day", "days", "week", "weeks", "wk", "wks", "month", "months", "year", "years",
                           "hour", "hours", "hr", "hrs", "h", "min", "mins", "minute", "minutes", "times",
                           "x", "doses", "dose")) | _UNIT_WORDS


def units_without_quantity(text: str) -> list[str]:
    """The units of the text that count (as in has_number_or_unit) but have no quantity of their own.

    A unit's quantity is the token right before it: a number with a real digit ("20 mg", "2.5 mg",
    ".5 mg", "1-2 tabs"), or a unit that has its quantity ("5 mmol/L", "1 tab od"); or the number is
    written together with the unit ("20mg"). Nothing else stands in for it: not a list number at the
    start of the line ("1. Omeprazole mg"), a duration ("Omeprazole mg for 7 days"), another medicine's
    dose ("Amlodipine 5 mg and Omeprazole mg") or any other digit on the line. Besides:
      - frequency words (od, bd, tds, qds, prn, nocte, mane) need no quantity ("Apply the cream bd");
      - a count unit (tab, cap, drop, puff, ...) also takes a number word or "a" ("Take one tablet at
        night"); its plural without a number names the form ("Paracetamol tablets as needed"); and a
        singular one that starts the line before a word is the form of the medicine ("tab Metformin 500 mg");
      - a dose or measurement unit written before its value, as in lab tables ("Glucose (mmol/L) 5.4"),
        takes the number right after it (after any units of its own), written in digits only, unless that
        number belongs to the word after it ("Omeprazole mg 7 days", "Omeprazole mg 1 tab od");
      - but a frequency word right after a capitalised name, with no dose before it, marks a lost dose
        ("Amitriptyline nocte", "Bisoprolol od"), while one after a form or a word does not ("Apply the
        cream bd", "Salbutamol inhaler prn"); and g, l, mol, iu, unit(s) count as units right before a
        unit or frequency word that counts ("Paracetamol g prn")."""
    chunks = (text or "").split()
    if chunks and _LIST_INDEX.fullmatch(chunks[0]):
        chunks = chunks[1:]
    parts = _parts(" ".join(chunks))
    counted = [False] * len(parts)
    prev_counts = False  # the previous token is number-like, so a unit after it counts
    for i, part in enumerate(parts):
        counted[i] = part.isalpha() and part.lower() in _UNIT_WORDS and (part in _USUAL_SPELLING or prev_counts)
        prev_counts = _number_like(part) or _is_quantity(part)
    for i in range(len(parts) - 2, -1, -1):
        if not counted[i] and counted[i + 1] and parts[i].lower() in _AMBIGUOUS_UNITS and not _word_like(parts[i]):
            counted[i] = True

    def value_after(i: int) -> bool:
        j = i + 1
        while j < len(parts) and parts[j].isalpha() and parts[j].lower() in _UNIT_WORDS:
            j += 1  # its own compound units: "mmol/L 5.4", "mg/dL 98"
        if j >= len(parts) or not _DIGITS.fullmatch(parts[j]):
            return False
        return j + 1 >= len(parts) or parts[j + 1].lower() not in _TAKES_NUMBER

    words = [p.lower() for p in parts if p.lower() not in _UNIT_WORDS]
    header_line = bool(words) and all(w in _HEADER_WORDS for w in words)  # "Analyte (mmol/L)"
    out = []
    prev_ok = False  # the previous token is a quantity, or a unit that has its quantity
    for i, part in enumerate(parts):
        if counted[i]:
            low = part.lower()
            if low in _FREQUENCY_WORDS:
                if not prev_ok and i and _word_like(parts[i - 1]) and not counted[i - 1]:
                    out.append(part)  # "Amitriptyline nocte": the dose between name and frequency is gone
                continue  # prev_ok carries over: "1 tab od"
            ok = prev_ok
            if not ok and low in _COUNT_UNITS:
                before = parts[i - 1].lower() if i else ""
                lost_count = (low in _COUNT_PLURALS and 0 < i < len(parts) - 1 and _word_like(parts[i - 1])
                              and before not in _FORM_NAMES and parts[i + 1].lower() in _FREQUENCY_WORDS
                              and counted[i + 1])  # "Salbutamol puffs prn"
                ok = (before in _COUNT_WORDS or (low in _COUNT_PLURALS and not lost_count)
                      or (i == 0 and i + 1 < len(parts) and parts[i + 1].isalpha() and not counted[i + 1]))
            if not ok and low not in _COUNT_UNITS:
                ok = value_after(i) or header_line
            if not ok:
                out.append(part)
            prev_ok = ok
            continue
        m = _QUANTITY_UNIT.fullmatch(part)
        attached = bool(m) and any(c.isdigit() for c in m.group(1)) and m.group(2).lower() in _UNIT_WORDS
        prev_ok = attached or _is_quantity(part)
    return out


def number_lost(text: str) -> bool:
    """True when a unit counts (see has_number_or_unit) but has no quantity of its own (units_without_quantity:
    "Omeprazole mg", "Insulin glargine IZ Units", "1. Omeprazole mg", "Omeprazole mg for 7 days",
    "Amlodipine 5 mg and Omeprazole mg"). The number that went with it was probably lost or misread; a
    digit elsewhere on the line does not stand in for it. Frequency words, count units with a number word
    and forms ("Take one tablet at night", "Eye drops nocte", "tab Metformin 500 mg") are not lost numbers."""
    return has_number_or_unit(text) and bool(units_without_quantity(text))


# Misread units that a printed reader makes of mg, mL and mcg ("Metformin SOO my").
_MISREAD_UNITS = frozenset(("my", "rng", "mq", "mI", "rnl", "rnL", "rncg", "mcq", "rnmol"))
_NOT_A_NUMBER = frozenset(("so", "is", "os", "sos", "oz"))  # words made of look-alike letters, in any case
# Dose units: a look-alike word in capitals right before one of these is a misread number after all
# ("Levothyroxine SO mcq": 50 mcg), while before a form or count it is the word ("Timolol OS drops": left eye).
_DOSE_UNITS = frozenset(("mcg", "µg", "ug", "mg", "g", "kg", "ml", "l", "dl", "mmol", "mol", "miu", "iu"))


def has_misread_number(text: str) -> bool:
    """True when a token of look-alike letters, with no real digit or with digits mixed with look-alike
    letters, stands right before a unit or a misread unit ("Metformin SOO my", "Atorvastatin ZO mg").
    Used only when Scanner(flag_misread_digits=True): off by default, see ocr/scan.py."""
    parts = _parts(text)
    for a, b in zip(parts, parts[1:]):
        word = a.lower() in _NOT_A_NUMBER and not (a.isupper() and (b in _MISREAD_UNITS or b.lower() in _DOSE_UNITS))
        if not _NUMBER_LIKE.fullmatch(a) or not any(c.isalpha() for c in a) or word or _word_like(a):
            continue
        if b in _MISREAD_UNITS or (b.isalpha() and b.lower() in _UNIT_WORDS):
            return True
    return False


def has_misread_unit(text: str) -> bool:
    """True when a number with real digits stands right before a misread unit: "Levothyroxine 50 mcq OD",
    "Lactulose 10 mI bd", "Haemoglobin 11.8 q/dL" (q for g before /dL or /L). Speculative, so it is used
    only with flag_misread_digits (off by default), like has_misread_number."""
    parts = _parts(text)
    for i in range(len(parts) - 1):
        a, b = parts[i], parts[i + 1]
        if not _is_quantity(a) or (b == "my" and re.fullmatch(r"(?:19|20)\d\d", a)):
            continue  # "Since 2019 my blood pressure": a year, then the word "my"
        if b in _MISREAD_UNITS:
            return True
        if b == "q" and i + 2 < len(parts) and parts[i + 2].lower() in ("dl", "l"):
            return True
    return False


READINGS = ("text", "handwriting_text", "printed_text")  # the shown reading first, then the other ones


def line_confirm_numbers(ln: dict, *, script_known: bool | None = None, flag_misread_digits: bool = False) -> bool:
    """Must a person check this line's numbers against the page, whatever its confidence? (the
    confirm_numbers flag of one line; ocr.scan.confirm_numbers is this function, torch-free here)

    ln holds the shown "text" and, when the line went to the handwriting reader, "handwriting_text" and
    (when the handwriting reading was chosen) "printed_text". Yes when:
      - the line went to the handwriting reader (or is handwritten) and any reading holds a number or unit:
        handwritten doses are misread with high confidence (2.5 mg read as 25 mg), and either reader can
        be the one that kept the digits;
      - the script is unknown (no printed / handwritten routing ran: script_known False; None means known
        when the line has a "script") and the text holds a number or unit, since it may be handwriting;
      - the shown text has a unit without its own number (number_lost: "Omeprazole mg", "1. Omeprazole mg",
        "Amlodipine 5 mg and Omeprazole mg");
      - flag_misread_digits is on and any reading has letters that look like misread digits before a unit
        ("Metformin SOO my", has_misread_number) or digits before a misread unit ("50 mcq",
        has_misread_unit). Off by default.
    A printed line with its numbers in place ("1. Omeprazole 20 mg") is not flagged here; it gets the
    quieter has_numbers flag."""
    readings = [ln.get(k) or "" for k in READINGS]
    if script_known is None:
        script_known = "script" in ln
    routed = "handwriting_text" in ln or ln.get("script") == "handwritten"
    if (not script_known or routed) and any(has_number_or_unit(t) for t in readings):
        return True
    if number_lost(readings[0]):
        return True
    return bool(flag_misread_digits) and any(has_misread_number(t) or has_misread_unit(t) for t in readings)

"""Deterministic drug / value / unit parser used to score readings against labelled ground truth.

parse_fields("1. Tab Amlodipine 5 mg OD and Omeprazole mg") ->
    [{"drug": "amlodipine", "value": "5", "unit": "mg"}, {"drug": "omeprazole", "value": None, "unit": "mg"}]

Rules (all of them; anything else is ignored):
- A field starts at a name from NAMES (medicines and lab analytes), matched case-insensitively on whole
  tokens, longest name first ("insulin glargine" before "insulin"). Between the name and the value only "to", ":",
  "-", "=" and "of" are skipped ("Increase Amlodipine to 10 mg").
- value: the first token after the name that is a number: digits with an optional decimal point or comma
  part (".5", "0.5", "2.5", "2,5", "1000"), kept as the exact string. A number glued to a unit ("500mg")
  is split. Ranges ("12.0-15.5") and "x10^9/L" after the value are not parsed.
- unit: the token right after the value (or right after the name when there is no value) when it is in
  UNITS, compared case-insensitively and returned in canonical lower case ("mL" and "ML" -> "ml").
  Misread units ("my", "mcq", "rng") are not units, so they give unit None.
- Unit before value (lab layouts "Glucose (mmol/L) 5.4", "Glucose mmol/L 5.4"): when the name is followed
  by a unit and then a number, that number is the value.
- The field ends at the next name; a value or unit after another word than those skipped is not taken
  ("Omeprazole before breakfast 20 mg" -> value None).

Known limits: names outside NAMES are not seen (a misread name such as "lnsulin" gives no field, which
counts as a disagreement); frequencies, durations and routes are not parsed; only the first value per name
is taken; units are compared without case, so "ML" for "mL" is not an error here.
"""
from __future__ import annotations

import re

NAMES = tuple(sorted({
    # medicines
    "paracetamol", "amlodipine", "metformin", "atorvastatin", "ramipril", "omeprazole", "levothyroxine",
    "amoxicillin", "salbutamol", "insulin", "insulin glargine", "warfarin", "prednisolone", "bisoprolol",
    "losartan", "cetirizine", "ibuprofen", "folic acid", "furosemide", "gliclazide", "sertraline", "morphine",
    "lorazepam", "digoxin", "amitriptyline", "lactulose", "vitamin d", "aspirin", "colecalciferol",
    "methotrexate", "enoxaparin", "clopidogrel", "simvastatin", "lisinopril", "doxycycline", "tramadol",
    "hydroxychloroquine",
    # lab analytes
    "haemoglobin", "fasting glucose", "glucose", "hba1c", "cholesterol", "total cholesterol", "creatinine",
    "sodium", "potassium", "tsh", "urea", "albumin", "ferritin", "vitamin b12", "platelets", "mcv", "haematocrit",
}, key=lambda s: (-len(s.split()), s)))
UNITS = frozenset((
    "mg", "mcg", "µg", "ug", "g", "kg", "ml", "l", "dl", "mmol", "mmol/l", "umol/l", "µmol/l", "g/dl", "g/l",
    "iu", "miu/l", "unit", "units", "%", "mmhg", "tab", "tabs", "tablet", "tablets", "cap", "caps", "capsule",
    "capsules", "puff", "puffs", "drop", "drops", "fl", "ug/l", "µg/l", "nmol/l", "pmol/l", "u/l", "mg/l", "mg/dl",
    "mmol/mol"))
SKIP_AFTER = frozenset(("to", ":", "-", "of", "="))  # between a name and its value
_NUM = re.compile(r"^(?:\d+(?:[.,]\d+)?|[.,]\d+)$")
_GLUED = re.compile(r"^(\d+(?:[.,]\d+)?|[.,]\d+)([A-Za-zµ%/]+)$")
_LIST = re.compile(r"^\(?\d{1,2}[.)]$")


def tokens(text: str) -> list[str]:
    out = []
    for t in (text or "").split():
        t = t.strip(",;\"'()[]")
        if t.endswith(".") and not _NUM.match(t) and not _LIST.match(t):
            t = t[:-1]
        if not t:
            continue
        m = _GLUED.match(t)
        if m and m.group(2).lower() in UNITS:
            out.extend([m.group(1), m.group(2)])
        else:
            out.append(t)
    return out


def parse_fields(text: str, *, preserve_unit_case: bool = False) -> list[dict]:
    toks = tokens(text)
    low = [t.lower() for t in toks]
    starts = []  # (index, name, length)
    i = 0
    while i < len(low):
        for name in NAMES:
            parts = name.split()
            if low[i:i + len(parts)] == parts:
                starts.append((i, name, len(parts)))
                i += len(parts) - 1
                break
        i += 1
    fields = []
    for k, (i, name, n) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(toks)
        j = i + n
        value = unit = None
        while j < end and low[j] in SKIP_AFTER:
            j += 1
        if j < end and _NUM.match(toks[j]):
            value = toks[j]
            j += 1
        if j < end and low[j] in UNITS:
            unit = toks[j] if preserve_unit_case else low[j]
            if value is None and j + 1 < end and _NUM.match(toks[j + 1]):  # "Glucose (mmol/L) 5.4"
                value = toks[j + 1]
        fields.append({"drug": name, "value": value, "unit": unit})
    return fields


def numbers(text: str) -> list[str]:
    """Every number in the text (exact strings, in order), including digits inside words ("2l/04" -> 2, 04)."""
    return re.findall(r"\d+(?:[.,]\d+)?|(?<![\w])[.,]\d+", text or "")


def units(text: str) -> list[str]:
    return [t.lower() for t in tokens(text) if t.lower() in UNITS]

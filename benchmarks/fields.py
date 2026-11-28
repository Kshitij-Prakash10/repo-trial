"""Deterministic drug / value / unit parser used to score readings against labelled ground truth.

parse_fields("1. Tab Amlodipine 5 mg OD and Omeprazole mg") ->
    [{"drug": "amlodipine", "value": "5", "unit": "mg"}, {"drug": "omeprazole", "value": None, "unit": "mg"}]

Rules (all of them; anything else is ignored):
- A field starts at a name from NAMES (medicines and lab analytes), matched case-insensitively on whole
  tokens, longest name first ("insulin glargine" before "insulin"). Between the name and the value only "to", ":",
  "-", "=" and "of" are skipped ("Increase Amlodipine to 10 mg").
- value: the first token after the name that is a number: digits with an optional decimal point or comma
  part (".5", ",5", "0.5", "2.5", "2,5", "1000"), kept as the exact string, including a leading decimal
  separator: ",5" stays ",5" (reading it as "5" is an error). A number glued to a unit ("500mg", "0.5mg")
  is split. Ranges ("12.0-15.5") and "x10^9/L" after the value are not parsed. Commas and semicolons
  that end a word ("500 mg, then") are punctuation and are dropped.
- unit: the token right after the value (or right after the name when there is no value) when it is in
  UNITS, compared case-insensitively and returned in canonical lower case ("mL" and "ML" -> "ml").
  Misread units ("my", "mcq", "rng") are not units, so they give unit None.
- Unit before value (lab layouts "Glucose (mmol/L) 5.4", "Glucose mmol/L 5.4"): when the name is followed
  by a unit and then a number, that number is the value.
- The field ends at the next name; a value or unit after another word than those skipped is not taken
  ("Omeprazole before breakfast 20 mg" -> value None).

Known limits: names outside NAMES are not seen (a misread name such as "lnsulin" gives no field, which
counts as a disagreement); frequencies, durations and routes are not parsed; only the first value after
each occurrence of a name is taken; units are compared without case, so "ML" for "mL" is not an error here.
The parser sees only the fixed name vocabulary in NAMES: field metrics say nothing about other words.

Decimal numbers (`is_decimal`, `has_decimal`): a number with a point or comma followed by digits (".5",
",5", "0,5", "2.5"). The one exception is a comma followed by exactly three digits after a non-zero whole
part ("1,000"), read as a thousands separator. Dates and fractions with slashes ("12/04/2026", "139/96")
hold no decimal. Every benchmark path uses these two functions, not a check for "." alone.

Field occurrences (`match_fields`, `field_match_scores`): every occurrence of a name is its own field, and
each predicted occurrence can match at most one reference occurrence. Exact matches (same name, value and
unit) are paired first, in reading order; the remaining occurrences of the same name are paired as
substitutions; what is left is an omission (reference) or an extra (prediction). See `match_fields`.
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


_LEADING_DECIMAL = re.compile(r"^[.,]\d")


def _clean(t: str) -> str:
    """Strip quotes, brackets and word-final commas / semicolons, but keep a leading decimal separator
    (",5" and ".5" are numbers, not punctuation followed by 5)."""
    while True:
        s = t.strip("\"'()[]").rstrip(",;")
        if s[:1] in (",", ";") and not _LEADING_DECIMAL.match(s):
            s = s[1:]
        if s == t:
            return s
        t = s


def tokens(text: str, *, legacy_punctuation: bool = False) -> list[str]:
    """Whitespace tokens with punctuation removed and numbers split from glued units. legacy_punctuation
    reproduces the tokenizer before the decimal fix (it dropped a leading decimal comma: ",5" -> "5"),
    only to rescore saved predictions with the legacy metrics."""
    out = []
    for t in (text or "").split():
        t = t.strip(",;\"'()[]") if legacy_punctuation else _clean(t)
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


def parse_fields(text: str, *, preserve_unit_case: bool = False, legacy_punctuation: bool = False) -> list[dict]:
    toks = tokens(text, legacy_punctuation=legacy_punctuation)
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


# ------------------------------------------------------------------ decimals
_DECIMAL = re.compile(r"^(\d*)[.,](\d+)$")


def is_decimal(number: str) -> bool:
    """True for a number written with a decimal separator: ".5", ",5", "0,5", "2.5". A comma followed by
    exactly three digits after a non-zero whole part ("1,000") is a thousands separator, not a decimal."""
    m = _DECIMAL.match(number or "")
    if not m:
        return False
    whole, frac = m.groups()
    return not ("," in number and whole.strip("0") and len(frac) == 3)


def has_decimal(text: str) -> bool:
    """True when any number in the text (see `numbers`) is a decimal (see `is_decimal`)."""
    return any(is_decimal(n) for n in numbers(text))


# ------------------------------------------------------------------ field occurrences
def _key(f: dict, unit_case: bool) -> tuple:
    unit = f.get("unit")
    if unit is not None and not unit_case:
        unit = unit.lower()
    return (f["drug"], f.get("value"), unit)


def _lcs(a: list, b: list) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def match_fields(pred: list[dict], ref: list[dict], *, unit_case: bool = False) -> dict:
    """Match field occurrences of one prediction against one reference (both `parse_fields` output, in
    reading order). Each occurrence is used at most once.

    1. exact: a reference occurrence takes the first unused predicted occurrence with the same name,
       value and unit (units compared without case unless unit_case).
    2. substitution: a remaining reference occurrence takes the first unused predicted occurrence with the
       same name (its value or unit differs).
    3. omission: a reference occurrence left unmatched; extra: a predicted occurrence left unmatched.

    The counts do not depend on order (swapping two occurrences of one name keeps both exact);
    `exact_in_order` (longest common subsequence of exact fields) does, so a swap is visible there.
    Returns counts plus "pairs": [(reference field, matched predicted field or None, kind)]."""
    P = [_key(f, unit_case) for f in pred]
    R = [_key(f, unit_case) for f in ref]
    used = [False] * len(P)
    match = [None] * len(R)
    kind = [None] * len(R)
    for i, r in enumerate(R):
        for j, p in enumerate(P):
            if not used[j] and p == r:
                used[j], match[i], kind[i] = True, j, "exact"
                break
    for i, r in enumerate(R):
        if match[i] is None:
            for j, p in enumerate(P):
                if not used[j] and p[0] == r[0]:
                    used[j], match[i], kind[i] = True, j, "substitution"
                    break
    pairs = [(ref[i], pred[match[i]] if match[i] is not None else None, kind[i] or "omission")
             for i in range(len(R))]
    matched = [(R[i], P[match[i]]) for i in range(len(R)) if match[i] is not None]
    return {"ref_fields": len(R), "pred_fields": len(P),
            "exact": kind.count("exact"), "substitutions": kind.count("substitution"),
            "omissions": kind.count(None), "extras": used.count(False),
            "exact_in_order": _lcs(R, P),
            "value_correct": sum(r[1] == p[1] for r, p in matched),
            "unit_correct": sum(r[2] == p[2] for r, p in matched),
            "pairs": pairs}


COUNT_KEYS = ("ref_fields", "pred_fields", "exact", "substitutions", "omissions", "extras", "exact_in_order",
              "value_correct", "unit_correct")


def _frac(num: int, den: int) -> dict:
    return {"num": int(num), "den": int(den), "rate": round(num / den, 4) if den else None}


def field_match_scores(pairs, *, unit_case: bool = False) -> dict:
    """Field occurrence scores over (prediction, reference) text pairs, e.g. aligned lines. Lines whose
    reference is empty (inserted output) still count: their fields are extras.

    Raw counts (COUNT_KEYS) plus fractions {"num", "den", "rate"}:
      precision = exact / predicted fields, recall = exact / reference fields,
      exact_in_order = order-sensitive exact / reference fields,
      drug, value, unit = matched name / correct value / correct unit (among matched names) per reference
      field, and "exact" = recall (same key and denominator as the earlier metric, with each occurrence
      matched at most once).
    A zero denominator gives rate None (no fields: precision or recall is undefined, not 0 or 1)."""
    tot = dict.fromkeys(COUNT_KEYS, 0)
    for pred, ref in pairs:
        m = match_fields(parse_fields(pred, preserve_unit_case=unit_case),
                         parse_fields(ref, preserve_unit_case=unit_case), unit_case=unit_case)
        for k in COUNT_KEYS:
            tot[k] += m[k]
    n, p = tot["ref_fields"], tot["pred_fields"]
    return {"counts": tot,
            "precision": _frac(tot["exact"], p), "recall": _frac(tot["exact"], n),
            "exact_in_order": _frac(tot["exact_in_order"], n),
            "drug": _frac(tot["exact"] + tot["substitutions"], n),
            "value": _frac(tot["value_correct"], n), "unit": _frac(tot["unit_correct"], n),
            "exact": _frac(tot["exact"], n)}

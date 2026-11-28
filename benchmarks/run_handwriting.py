"""Handwriting diagnosis benchmark: the shipped readers and the shipped selection rule on annotated lines.

usage: PYTHONPATH=<dir with transformers 4.57> HF_HUB_OFFLINE=1 \
       python -m benchmarks.run_handwriting --root <checkout> --out <dir> [--limit N]

Data (deterministic, invented, held out): multi-line handwritten medical pages rendered here with the
repository's open-licence handwriting fonts (data/raw/fonts, the ones ocr/synth_hw.py uses), word by word,
so every word and line has a known box and text. Doses with decimals, frequencies and numbered lists are
included. Pages with seeds 0..19 are the "validation" part (used to choose the candidate selector's margin),
seeds 1000..1029 the "test" part (evaluated once). The handwriting reader was fine-tuned on lines rendered
with these same fonts (ocr/synth_hw.py), so the results are OPTIMISTIC for real handwriting; real held-out
pages (GNHK, IAM) need huggingface.co and are BLOCKED here.

Per annotated line, as ocr/scan.py does it:
  printed reading: the printed recogniser (models/ocr/rec_crnn_mobilenet_v3_large_ft.pt) on each word box,
      joined with spaces; confidence = mean of the word confidences (OcrPipeline.run);
  routing: ocr.scriptclf P(handwritten) per word crop, the line goes to the handwriting reader when the mean
      reaches ocr.scan.HW_LINE_SHARE;
  handwriting reading: ocr.handwriting.HandwritingReader on ocr.hwdata.rect_crop of the word quads;
  shipped rule: keep_printed() first, then the handwriting reading replaces the printed one only when its
      confidence is higher (Scanner.run). "shipped" applies routing too; "shipped_if_routed" assumes routing.
The word boxes are the annotated ones, not detector boxes: this isolates recognition and selection from
line detection. Automatic line detection is scored by match_lines() when docTR's text detector weights are
present (models/ocr/det_fast_base_ft.pt or ~/.cache/doctr/models/fast_base-*.pt); otherwise BLOCKED.
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import common as C  # noqa: E402
from benchmarks.fields import field_match_scores, has_decimal, numbers, parse_fields  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

VAL_SEEDS = tuple(range(0, 20))
TEST_SEEDS = tuple(range(1000, 1030))
THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))  # 0.50 .. 0.95
BINS = tuple(round(0.1 * i, 1) for i in range(11))
MARGINS = tuple(round(-0.3 + 0.05 * i, 2) for i in range(13))  # candidate selector grid, chosen on validation

DRUGS = ("Ramipril", "Metformin", "Amlodipine", "Omeprazole", "Atorvastatin", "Levothyroxine", "Lorazepam",
         "Warfarin", "Bisoprolol", "Prednisolone", "Paracetamol", "Amoxicillin", "Insulin", "Furosemide",
         "Sertraline", "Digoxin", "Lactulose", "Salbutamol", "Gliclazide", "Aspirin")
DOSES = {"mg": ("2.5", "5", "10", "20", "40", "50", "75", "500", "0.5", "1.25", "12.5", "100"),
         "mcg": ("25", "50", "62.5", "75", "100", "125"), "mL": ("5", "10", "15", "2.5"),
         "units": ("4", "8", "10", "12", "16", "20"), "g": ("1", "0.5")}
FREQ = ("OD", "BD", "TDS", "nocte", "prn", "once daily", "twice daily", "at night", "after meals", "")
TEXT = ("Continue same medicines", "Review in clinic", "Dr A. Example", "Take with food", "See me if worse",
        "Low salt diet", "Good progress", "Stop if rash", "Check sugar weekly", "Signed")


def make_line(rng) -> str:
    k = rng.random()
    if k < 0.62:
        drug = str(rng.choice(DRUGS))
        unit = {"Insulin": "units", "Levothyroxine": "mcg", "Digoxin": "mcg", "Lactulose": "mL",
                "Paracetamol": str(rng.choice(["g", "mg"]))}.get(drug, "mg")
        s = f"{drug} {rng.choice(DOSES[unit])} {unit} {rng.choice(FREQ)}".strip()
        if rng.random() < 0.3:
            s = f"{int(rng.integers(1, 6))}. {s}"
        return s
    if k < 0.75:
        return f"Review on {int(rng.integers(1, 29)):02d}/{int(rng.integers(1, 13)):02d}/2026"
    if k < 0.82:
        return f"BP {int(rng.integers(100, 170))}/{int(rng.integers(60, 100))}"
    return str(rng.choice(TEXT))


def font_files(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "data", "raw", "fonts", "*.ttf")))


def render_page(seed: int, fonts: list[str]) -> dict:
    """{"image" (RGB uint8), "lines": [{"text", "quad" (4, 2), "words": [{"text", "quad"}]}]}."""
    rng = np.random.default_rng([C.SEED, 31, seed])
    font = ImageFont.truetype(fonts[int(rng.integers(len(fonts)))], int(rng.integers(40, 58)))
    W = 1400
    n = int(rng.integers(4, 8))
    texts = [make_line(rng) for _ in range(n)]
    gap = int(font.size * rng.uniform(1.5, 1.9))
    H = 120 + gap * n + 80
    ink = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(ink)
    lines = []
    y = 90
    for t in texts:
        x = int(rng.integers(60, 160))
        words = []
        for w in t.split():
            x0, y0, x1, y1 = d.textbbox((x, y), w, font=font)
            d.text((x, y), w, fill=255, font=font)
            words.append({"text": w, "box": [x0 - 3, y0 - 3, x1 + 3, y1 + 3]})
            x = x1 + int(font.size * rng.uniform(0.3, 0.6))
        b = np.array([w["box"] for w in words], float)
        lines.append({"text": t, "words": words, "box": [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]})
        y += gap + int(rng.integers(-4, 5))
    # a small page tilt, the same for ink and boxes
    ang = float(rng.uniform(-1.5, 1.5))
    t = np.deg2rad(ang)
    c = np.array([W / 2, H / 2])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    ink = ink.rotate(-ang, resample=Image.BILINEAR, center=tuple(c))  # PIL turns counter-clockwise for +angle
    def quad(box):
        x0, y0, x1, y1 = box
        p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
        return (p - c) @ R.T + c
    for ln in lines:
        ln["quad"] = quad(ln["box"])
        for w in ln["words"]:
            w["quad"] = quad(w["box"])
        del ln["box"]
        for w in ln["words"]:
            del w["box"]
    a = np.asarray(ink, np.float32) / 255.0 * rng.uniform(0.75, 1.0)
    paper = np.full((H, W, 3), rng.uniform(215, 250), np.float32) * rng.uniform(0.94, 1.0, 3)
    colour = np.array([[20, 20, 30], [15, 25, 110], [40, 40, 45]][int(rng.integers(3))], np.float32)
    img = paper * (1 - a[..., None]) + colour * a[..., None] + rng.normal(0, rng.uniform(1, 5), (H, W, 1))
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=int(rng.integers(55, 91)))
    buf.seek(0)
    return {"seed": seed, "image": np.asarray(Image.open(buf).convert("RGB")), "lines": lines}


# ------------------------------------------------------------------ metrics
def edit(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer_wer(pairs) -> dict:
    ce = sum(edit(p, t) for p, t in pairs)
    cn = sum(len(t) for _, t in pairs)
    we = sum(edit(p.split(), t.split()) for p, t in pairs)
    wn = sum(len(t.split()) for _, t in pairs)
    return {"cer": round(ce / cn, 4) if cn else None, "wer": round(we / wn, 4) if wn else None,
            "char_errors": ce, "chars": cn, "word_errors": we, "words": wn, "n": len(pairs)}


def norm(s: str) -> str:
    return " ".join((s or "").split())


def field_scores(pairs) -> dict:
    """Field occurrence scores (benchmarks.fields.field_match_scores): each predicted occurrence matches at
    most one reference occurrence; extras, omissions and substitutions are counted, with precision and
    recall. "exact", "drug", "value" and "unit" keep their earlier names and reference-field denominators."""
    return field_match_scores(pairs)


def field_scores_legacy(pairs) -> dict:
    """The metric before the occurrence matcher, kept to rescore saved predictions: predicted fields are
    looked up by name, so a repeated name collapses to one prediction that can be credited to every
    reference occurrence, and extra predicted fields are never counted."""
    tot = {"fields": 0, "drug": 0, "value": 0, "unit": 0, "exact": 0}
    for pred, ref in pairs:
        pf = {f["drug"]: f for f in parse_fields(pred, legacy_punctuation=True)}
        for f in parse_fields(ref, legacy_punctuation=True):
            tot["fields"] += 1
            g = pf.get(f["drug"])
            if g is None:
                continue
            tot["drug"] += 1
            tot["value"] += g["value"] == f["value"]
            tot["unit"] += g["unit"] == f["unit"]
            tot["exact"] += g == f
    n = tot.pop("fields")
    return {k: C.frac(v, n) for k, v in tot.items()}


def numeric_line(pairs) -> dict:
    sub = [(p, t) for p, t in pairs if re.search(r"\d", t)]
    return C.frac(sum(norm(p) == norm(t) for p, t in sub), len(sub))


def wrong_decimal(pred: str, ref: str) -> bool:
    """The reference holds a decimal number (benchmarks.fields.is_decimal: point or comma) and the reading
    does not have the same numbers, written exactly the same way."""
    return has_decimal(ref) and numbers(pred) != numbers(ref)


def calibration(rows, key_text, key_conf) -> dict:
    table = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sub = [r for r in rows if lo <= r[key_conf] < hi or (hi == 1.0 and r[key_conf] == 1.0)]
        if sub:
            table.append({"bin": f"[{lo:.1f}, {hi:.1f})", "n": len(sub),
                          "mean_confidence": round(float(np.mean([r[key_conf] for r in sub])), 4),
                          "line_exact": C.frac(sum(norm(r[key_text]) == norm(r["truth"]) for r in sub), len(sub))})
    acc = {}
    for t in THRESHOLDS:
        sub = [r for r in rows if r[key_conf] >= t]
        acc[f"{t:.2f}"] = {
            "accepted": C.frac(len(sub), len(rows)),
            "errors_among_accepted (line not exact)": C.frac(sum(norm(r[key_text]) != norm(r["truth"]) for r in sub), len(sub)),
            "number_errors_among_accepted": C.frac(sum(numbers(r[key_text]) != numbers(r["truth"]) for r in sub
                                                       if re.search(r"\d", r["truth"])),
                                                   sum(bool(re.search(r"\d", r["truth"])) for r in sub)),
            "wrong_decimals_among_accepted": C.frac(sum(wrong_decimal(r[key_text], r["truth"]) for r in sub),
                                                    sum(has_decimal(r["truth"]) for r in sub)),
            "wrong_decimal_examples": [f"{r['truth']} -> {r[key_text]} ({r[key_conf]:.2f})" for r in sub
                                       if wrong_decimal(r[key_text], r["truth"])][:8]}
    return {"reliability": table, "by_threshold": acc}


# ------------------------------------------------------------------ automatic line detection (needs the detector)
def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def match_lines(pred: list, true: list, iou_thr: float = 0.5, part: float = 0.5) -> dict:
    """Compare predicted line boxes (reading order) with annotated ones (reading order); boxes x0 y0 x1 y1.

    matched: one-to-one IoU >= iou_thr (best first). Of the rest: a true line is "split" when >= 2 unmatched
    predicted lines lie at least `part` inside it and together cover at least `part` of it; a predicted line
    is "merged" when >= 2 unmatched true lines lie at least `part` inside it. Remaining true lines are
    "missed", remaining predicted lines "extra". "misordered": matched lines not on the longest run that keeps
    the annotated order (len(matched) - longest increasing subsequence); "inversions": pairs out of order."""
    P, T = [list(map(float, b)) for b in pred], [list(map(float, b)) for b in true]
    cand = sorted(((_inter(p, t) / max(_area(p) + _area(t) - _inter(p, t), 1e-9), i, j)
                   for i, p in enumerate(P) for j, t in enumerate(T)), key=lambda x: (-x[0], x[1], x[2]))
    mp, mt, pairs = set(), set(), []
    for iou, i, j in cand:
        if iou < iou_thr:
            break
        if i not in mp and j not in mt:
            mp.add(i), mt.add(j), pairs.append((i, j))
    split_t, split_p = set(), set()
    for j, t in enumerate(T):
        if j in mt:
            continue
        parts = [i for i, p in enumerate(P) if i not in mp and _inter(p, t) >= part * max(_area(p), 1e-9)]
        if len(parts) >= 2 and sum(_inter(P[i], t) for i in parts) >= part * _area(t):
            split_t.add(j), split_p.update(parts)
    merged_p, merged_t = set(), set()
    for i, p in enumerate(P):
        if i in mp or i in split_p:
            continue
        inside = [j for j, t in enumerate(T) if j not in mt and j not in split_t and _inter(p, t) >= part * max(_area(t), 1e-9)]
        if len(inside) >= 2:
            merged_p.add(i), merged_t.update(inside)
    seq = [j for i, j in sorted(pairs)]
    lis = []
    for v in seq:
        k = int(np.searchsorted(lis, v))
        lis[k:k + 1] = [v]
    inv = sum(1 for a in range(len(seq)) for b in range(a + 1, len(seq)) if seq[a] > seq[b])
    return {"true": len(T), "pred": len(P), "matched": len(pairs), "split": len(split_t), "merged": len(merged_p),
            "missed": len(T) - len(mt) - len(split_t) - len(merged_t),
            "extra": len(P) - len(mp) - len(split_p) - len(merged_p),
            "misordered": len(seq) - len(lis), "inversions": inv}


def detected_lines(detect, group_lines, union, image) -> list:
    """Line boxes in reading order from a word detector: detect(image) -> (N, 4) word boxes (px)."""
    boxes = np.asarray(detect(image), float).reshape(-1, 4)
    return [union(boxes[ln]) for ln in group_lines(boxes)] if len(boxes) else []


def detector_prerequisite(root: str, det_ckpt: str) -> str | None:
    cache = os.environ.get("DOCTR_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "doctr")
    if os.path.exists(det_ckpt) or glob.glob(os.path.join(cache, "models", "fast_base-*.pt")):
        return None
    return (f"BLOCKED: docTR fast_base text-detector weights are missing: neither {os.path.relpath(det_ckpt, root)} "
            f"nor {os.path.join(cache, 'models', 'fast_base-*.pt')} exists (download host doctr-static.mindee.com "
            "is blocked here)")


# ------------------------------------------------------------------ decoding options passed through to the reader
def split_hw_options(options: list[str] | None, reader_cls) -> dict:
    """--hw-option key=value pairs -> keyword arguments for HandwritingReader(...) or its read(...).

    A key in HandwritingReader.__init__ goes to the constructor, else a key in read() goes to every read
    call, else it is ignored (the checkout lacks the option) and recorded as such. Values are parsed as JSON
    ("4" -> 4, "true" -> True), otherwise kept as text. With no options both calls are exactly as before."""
    import inspect
    import json
    init_p = set(inspect.signature(reader_cls.__init__).parameters) - {"self"}
    read_p = set(inspect.signature(reader_cls.read).parameters) - {"self", "lines"}
    out = {"requested": {}, "constructor": {}, "read": {}, "ignored": {}}
    for opt in options or []:
        if "=" not in opt:
            raise SystemExit(f"--hw-option needs key=value, got {opt!r}")
        k, v = opt.split("=", 1)
        try:
            val = json.loads(v)
        except ValueError:
            val = v
        out["requested"][k] = val
        out["constructor" if k in init_p else "read" if k in read_p else "ignored"][k] = val
    return out


# ------------------------------------------------------------------ main evaluation
def read_pages(pages, mods, latency):
    torch, models, scriptclf, hwdata, oscan, reader, rec = (mods[k] for k in
                                                           ("torch", "models", "scriptclf", "hwdata", "scan", "hw", "rec"))
    rows = []
    for pg in pages:
        img = pg["image"]
        for li, ln in enumerate(pg["lines"]):
            quads = [w["quad"] for w in ln["words"]]
            boxes = [np.r_[q.min(0), q.max(0)] for q in quads]
            crops = [img[max(0, int(y0)):int(np.ceil(y1)), max(0, int(x0)):int(np.ceil(x1))] for x0, y0, x1, y1 in boxes]
            t0 = time.perf_counter()
            pr = [(t, float(c)) for t, c in rec(crops)]
            t1 = time.perf_counter()
            p_hw = scriptclf.handwritten_prob(mods["clf"], crops)
            t2 = time.perf_counter()
            crop = hwdata.rect_crop(img, np.concatenate(quads))
            (hw_text, hw_conf), = reader.read([crop], **mods.get("read_kw", {}))
            t3 = time.perf_counter()
            latency["printed_ms"].append((t1 - t0) * 1000)
            latency["route_ms"].append((t2 - t1) * 1000)
            latency["handwriting_ms"].append((t3 - t2) * 1000)
            printed = " ".join(t for t, _ in pr)
            # The same decision functions as Scanner.run (ocr.scan), with the printed line confidence
            # computed as the pipeline reports it (ocr.pipeline.line_confidence: rounded word confidences).
            p_conf = oscan.line_confidence([c for _, c in pr]) if pr else 0.0
            share = oscan.route_share(p_hw)
            choose_hw, why = oscan.choose_reading(printed, p_conf, hw_text, hw_conf, crop)
            keep = bool(why)
            routed = oscan.is_routed(share)
            rows.append({"page": pg["seed"], "line": li, "truth": ln["text"], "printed": printed,
                         "printed_conf": round(p_conf, 4), "handwriting": hw_text, "handwriting_conf": round(hw_conf, 4),
                         "hw_share": round(share, 4), "routed": routed, "keep_printed": keep,
                         "shipped_if_routed": hw_text if choose_hw else printed,
                         "shipped_if_routed_conf": round(hw_conf if choose_hw else p_conf, 4),
                         "shipped": hw_text if (routed and choose_hw) else printed,
                         "shipped_conf": round(hw_conf if (routed and choose_hw) else p_conf, 4)})
    return rows


def candidate(rows, margin):
    """Candidate selector: handwriting reading when its confidence beats the printed one by more than
    `margin` (keep_printed still applies; routing assumed)."""
    out = []
    for r in rows:
        hw = (not r["keep_printed"]) and r["handwriting_conf"] - r["printed_conf"] > margin
        out.append(r["handwriting"] if hw else r["printed"])
    return out


def score(rows, texts=None) -> dict:
    keys = ("printed", "handwriting", "shipped", "shipped_if_routed")
    out = {}
    for k in keys + (("candidate",) if texts is not None else ()):
        preds = [r[k] for r in rows] if k != "candidate" else texts
        pairs = list(zip(preds, [r["truth"] for r in rows]))
        pages = {}
        for (p, t), r in zip(pairs, rows):
            pages.setdefault(r["page"], ([], []))
            pages[r["page"]][0].append(p)
            pages[r["page"]][1].append(t)
        out[k] = {"line": cer_wer(pairs), "page": cer_wer([("\n".join(a), "\n".join(b)) for a, b in pages.values()]),
                  "numeric_line_accuracy": numeric_line(pairs), "fields": field_scores(pairs),
                  "fields_legacy": field_scores_legacy(pairs),
                  "line_exact": C.frac(sum(norm(p) == norm(t) for p, t in pairs), len(pairs)),
                  "line_exact_ignoring_case": C.frac(sum(norm(p).lower() == norm(t).lower() for p, t in pairs), len(pairs))}
    return out


def diagnosis(rows) -> dict:
    n = len(rows)
    dis = sum(norm(r["printed"]) != norm(r["handwriting"]) for r in rows)
    num_dis = sum(numbers(r["printed"]) != numbers(r["handwriting"]) for r in rows)
    hw_ok = [norm(r["handwriting"]) == norm(r["truth"]) for r in rows]
    pr_ok = [norm(r["printed"]) == norm(r["truth"]) for r in rows]
    sh_ok = [norm(r["shipped"]) == norm(r["truth"]) for r in rows]
    oracle = sum(a or b for a, b in zip(hw_ok, pr_ok))
    return {"reader_disagreement (text)": C.frac(dis, n), "reader_disagreement (numbers)": C.frac(num_dis, n),
            "routed_to_handwriting": C.frac(sum(r["routed"] for r in rows), n),
            "lines_exact_by_either_reader (oracle selection)": C.frac(oracle, n),
            "lines_exact_shipped": C.frac(sum(sh_ok), n),
            "selection_loss (one reader right, shipped wrong)": C.frac(sum((a or b) and not s for a, b, s in
                                                                           zip(hw_ok, pr_ok, sh_ok)), n),
            "recognition_loss (both readers wrong)": C.frac(sum(not (a or b) for a, b in zip(hw_ok, pr_ok)), n),
            "routing_loss (handwriting right, line not routed)": C.frac(sum(a and not r["routed"] and not s for a, r, s in
                                                                            zip(hw_ok, rows, sh_ok)), n)}


def run(root, limit=None, threads=2, hw_options=None) -> dict:
    root = C.use_root(root)
    import torch  # noqa: E402
    torch.set_num_threads(threads)
    torch.manual_seed(C.SEED)
    from ocr import common, hwdata, models, scriptclf  # noqa: E402
    from ocr import scan as oscan  # noqa: E402
    from ocr.handwriting import HandwritingReader  # noqa: E402
    for m in (common, oscan):
        C.check_imported_from(m, root)
    fonts = font_files(root)
    if not fonts:
        raise SystemExit("no handwriting fonts in data/raw/fonts")
    val_seeds, test_seeds = (VAL_SEEDS[:limit], TEST_SEEDS[:limit]) if limit else (VAL_SEEDS, TEST_SEEDS)
    hw_opts = split_hw_options(hw_options, HandwritingReader)
    mods = {"torch": torch, "models": models, "scriptclf": scriptclf, "hwdata": hwdata, "scan": oscan,
            "hw": HandwritingReader(**{"local_only": True, **hw_opts["constructor"]}), "read_kw": hw_opts["read"], "rec": models.rec_predictor(models.load_rec(ckpt=models.REC_CKPT)),
            "clf": scriptclf.load_clf()}
    latency = {"printed_ms": [], "route_ms": [], "handwriting_ms": []}
    res = {"benchmark": "handwriting", "root": root, "git": C.git_info(root),
           "data": {"kind": "synthetic, invented text, rendered with data/raw/fonts (seen in training: optimistic)",
                    "fonts": len(fonts), "validation_seeds": [val_seeds[0], val_seeds[-1]],
                    "test_seeds": [test_seeds[0], test_seeds[-1]]},
           "readers": {"handwriting": mods["hw"].name, "printed": os.path.relpath(models.REC_CKPT, root)},
           "hw_options": hw_opts, "threads": threads,
           "real_heldout_pages": "BLOCKED: GNHK / IAM need huggingface.co (blocked here)"}
    parts = {}
    for name, seeds in (("validation", val_seeds), ("test", test_seeds)):
        pages = [render_page(s, fonts) for s in seeds]
        parts[name] = read_pages(pages, mods, latency)
    val, test = parts["validation"], parts["test"]
    grid = {f"{m:+.2f}": cer_wer(list(zip(candidate(val, m), [r["truth"] for r in val])))["cer"] for m in MARGINS}
    best = min(MARGINS, key=lambda m: (grid[f"{m:+.2f}"], abs(m)))
    res["candidate_selector"] = {
        "rule": "handwriting reading when its confidence exceeds the printed one by more than the margin "
                "(keep_printed still applies, routing assumed); margin chosen on validation by line CER",
        "validation_cer_by_margin": grid, "chosen_margin": best, "shipped_margin_equivalent": 0.0}
    for name, rows in parts.items():
        res[name] = {"n_pages": len({r["page"] for r in rows}), "n_lines": len(rows),
                     "scores": score(rows, candidate(rows, best)), "diagnosis": diagnosis(rows),
                     "calibration": {"handwriting_reader": calibration(rows, "handwriting", "handwriting_conf"),
                                     "shipped": calibration(rows, "shipped", "shipped_conf")},
                     "rows": rows}
    res["latency"] = {k: {"mean_ms": round(float(np.mean(v)), 1), "median_ms": round(float(np.median(v)), 1), "n": len(v)}
                      for k, v in latency.items()}
    pre = detector_prerequisite(root, common.DET_CKPT)
    if pre:
        res["detected_lines"] = {"status": pre}
    else:  # runs once the weights exist
        det = models.det_predictor(models.load_det(ckpt=common.DET_CKPT if os.path.exists(common.DET_CKPT) else None))
        def detect(img):
            out = det([img])[0]["words"]
            H, W = img.shape[:2]
            return np.clip(out[:, :4] * np.array([W, H, W, H]), 0, [W, H, W, H])
        tot = {}
        for s in test_seeds:
            pg = render_page(s, fonts)
            true = [np.r_[ln["quad"].min(0), ln["quad"].max(0)].tolist() for ln in pg["lines"]]
            m = match_lines(detected_lines(detect, common.group_lines, common.union, pg["image"]), true)
            for k, v in m.items():
                tot[k] = tot.get(k, 0) + v
        res["detected_lines"] = {"status": "ran", "test_totals": tot}
    return res


def markdown(res) -> str:
    L = [f"# Handwriting benchmark: {res['git'].get('commit') or res['root']}", "",
         f"Data: {res['data']['kind']}. Real held-out pages: {res['real_heldout_pages']}.", "",
         f"Automatic line detection: {res['detected_lines'].get('status')}.", "",
         f"Handwriting decoding options: {res.get('hw_options')}.", ""]
    for part in ("validation", "test"):
        p = res[part]
        L += [f"## {part}: {p['n_pages']} pages, {p['n_lines']} lines", ""]
        rows = []
        for k, v in p["scores"].items():
            rows.append([k, f"{v['line']['cer']} / {v['line']['wer']}", f"{v['page']['cer']} / {v['page']['wer']}",
                         v["line_exact"], v["line_exact_ignoring_case"], v["numeric_line_accuracy"], v["fields"]["value"], v["fields"]["unit"],
                         v["fields"]["exact"]])
        L += [C.md_table(["reading", "line CER / WER", "page CER / WER", "line exact", "exact ignoring case",
                          "numeric lines exact",
                          "field value", "field unit", "field exact"], rows), ""]
        L += [C.md_table(["diagnosis", "n"], [[k, v] for k, v in p["diagnosis"].items()]), ""]
        cal = p["calibration"]["handwriting_reader"]["by_threshold"]
        L += [C.md_table(["handwriting conf >=", "accepted", "errors among accepted", "number errors", "wrong decimals"],
                         [[t, v["accepted"], v["errors_among_accepted (line not exact)"], v["number_errors_among_accepted"],
                           v["wrong_decimals_among_accepted"]] for t, v in cal.items()]), ""]
    cs = res["candidate_selector"]
    L += [f"Candidate selector: {cs['rule']}; chosen margin {cs['chosen_margin']:+.2f}.", "",
          "Latency per line (ms, mean): " + ", ".join(f"{k} {v['mean_ms']}" for k, v in res["latency"].items())]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="pages per part")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hw-option", action="append", default=[], metavar="KEY=VALUE",
                    help="decoding option for HandwritingReader(...) or .read(...), e.g. num_beams=4; ignored "
                         "(and recorded) when the checkout lacks it")
    ap.add_argument("--name", default="handwriting", help="result file name")
    a = ap.parse_args(argv)
    C.limit_threads(a.threads)
    res = run(a.root, a.limit, a.threads, a.hw_option)
    print(C.write_result(a.out, C.out_name(a.name, a.limit), res, markdown(res)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

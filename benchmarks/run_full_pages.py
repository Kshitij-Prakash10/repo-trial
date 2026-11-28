"""Pair the unmodified Scanner.run image path with annotated-crop OCR on the same synthetic pages."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from benchmarks import common as C
from benchmarks import run_handwriting as hw
from benchmarks.fields import field_match_scores, is_decimal, match_fields, parse_fields


def align(reference, prediction):
    """Maximise exact line anchors in order, then minimise edit cost between anchors."""
    n, m = len(reference), len(prediction)
    scores = {(0, 0): (0, 0)}
    back = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if not (i or j):
                continue
            options = []
            if i and j:
                anchors, cost = scores[i - 1, j - 1]
                ref, pred = reference[i - 1], prediction[j - 1]
                options.append(((anchors - (ref == pred), cost + hw.edit(pred, ref)), (i - 1, j - 1)))
            if i:
                anchors, cost = scores[i - 1, j]
                options.append(((anchors, cost + len(reference[i - 1]) + 1), (i - 1, j)))
            if j:
                anchors, cost = scores[i, j - 1]
                options.append(((anchors, cost + len(prediction[j - 1]) + 1), (i, j - 1)))
            scores[i, j], back[i, j] = min(options, key=lambda item: item[0])
    pairs = []
    i, j = n, m
    while i or j:
        pi, pj = back[i, j]
        pairs.append((pi if i > pi else None, pj if j > pj else None))
        i, j = pi, pj
    return pairs[::-1]


def bounds(points):
    points = np.asarray(points).reshape(-1, 2)
    return [*points.min(axis=0), *points.max(axis=0)]


def rotated_bounds(quad, width, height, turns):
    # Scanner word source_quads refer to its upright input (including the page warp inverse).
    x, y = np.asarray(quad).T
    for _ in range(turns % 4):
        x, y = height - y, x
        width, height = height, width
    return bounds(np.column_stack([x, y]))


def score(pages, key):
    line_pairs, page_pairs = [], []
    for pg in pages:
        refs, preds = pg["reference"], pg[key]
        indices = pg["alignment"] if key == "detected" else enumerate(range(len(refs)))
        line_pairs.extend((preds[j] if j is not None else "", refs[i] if i is not None else "")
                          for i, j in indices)
        page_pairs.append((hw.norm(" ".join(preds)), hw.norm(" ".join(refs))))
    refs = [(p, r) for p, r in line_pairs if r]  # line-level metrics: reference lines only
    # Field metrics use every aligned pair, so fields on inserted output lines count as extras. Each
    # predicted occurrence matches at most one reference occurrence (benchmarks.fields.match_fields).
    decimals = []
    case_correct = missing = incomplete = 0
    for pred, ref in line_pairs:
        m = match_fields(parse_fields(pred, preserve_unit_case=True), parse_fields(ref, preserve_unit_case=True))
        for f, got, kind in m["pairs"]:
            missing += kind == "omission"
            incomplete += got is not None and (got["value"] is None or got["unit"] is None)
            case_correct += got is not None and got["unit"] == f["unit"]
            if is_decimal(f["value"] or ""):
                decimals.append(got is not None and got["value"] == f["value"])
    fields = field_match_scores(line_pairs)
    return {"page": hw.cer_wer(page_pairs), "line": hw.cer_wer(line_pairs),
            "exact_pages": C.frac(sum(p == r for p, r in page_pairs), len(pages)),
            "exact_lines": C.frac(sum(hw.norm(p) == hw.norm(r) for p, r in refs), len(refs)),
            "numeric_lines": hw.numeric_line(refs), "fields": fields,
            "case_sensitive_unit": C.frac(case_correct, fields["counts"]["ref_fields"]), "missing_drug": missing,
            "incomplete_fields_with_recognised_drug": incomplete,
            "exact_decimal_value": C.frac(sum(decimals), len(decimals)),
            "legacy": legacy_field_metrics(refs)}


def legacy_field_metrics(refs):
    """The field metrics before the occurrence matcher (kept to rescore saved predictions): reference lines
    only, predicted fields looked up by name (a repeated name collapses), decimals by "." only."""
    decimals = []
    case_correct = missing = incomplete = fields = 0
    for pred, ref in refs:
        by_name = {f["drug"]: f for f in parse_fields(pred, preserve_unit_case=True, legacy_punctuation=True)}
        for f in parse_fields(ref, preserve_unit_case=True, legacy_punctuation=True):
            fields += 1
            got = by_name.get(f["drug"], {})
            missing += not got
            incomplete += bool(got) and (got["value"] is None or got["unit"] is None)
            case_correct += bool(got) and got["unit"] == f["unit"]
            if "." in (f["value"] or ""):
                decimals.append(got.get("value") == f["value"])
    return {"fields": hw.field_scores_legacy(refs),
            "case_sensitive_unit": C.frac(case_correct, fields), "missing_drug": missing,
            "incomplete_fields_with_recognised_drug": incomplete,
            "exact_decimal_value": C.frac(sum(decimals), len(decimals))}


def paired_outcomes(pages):
    """Count changes associated with providing geometry; these are observations, not causal attributions."""
    counts = Counter()
    for pg in pages:
        for i, j in pg["alignment"]:
            if i is None:
                counts["inserted_output_line"] += 1
                continue
            ref = pg["reference"][i]
            crop = pg["annotated"][i]
            detected = pg["detected"][j] if j is not None else ""
            if detected == ref:
                key = "exact_both" if crop == ref else "exact_full_page_only"
            elif crop == ref:
                key = "exact_annotated_only"
            else:
                key = "wrong_both_same_text" if crop == detected else "wrong_both_different_text"
            counts[key] += 1
    return dict(counts)


# What the app sends: js/ocr.js posts only `image` (plus deskew=1 when "Straighten tilted text" is ticked),
# so /api/ocr reads with the page-scanner stage off (ocr/api.py read_page(use_scanner=False)).
APP_SETTINGS = {"page_scanner": False, "deskew": False}


def scanner_settings(page_scanner: bool, deskew: bool) -> dict:
    """The flags Scanner.run is given here, in the form of the /api/ocr request that reads the same way."""
    return {"page_scanner": bool(page_scanner), "deskew": bool(deskew),
            "api_request": {"scanner": "1" if page_scanner else "0", "deskew": "1" if deskew else "0"},
            "matches_app_default": {"page_scanner": bool(page_scanner), "deskew": bool(deskew)} == APP_SETTINGS}


def run(root, seeds, threads, page_scanner=True, deskew=False):
    root = Path(C.use_root(root))
    import torch
    from ocr import common, hwdata, models, scan, scriptclf
    from ocr.handwriting import HW_DIR
    from ocr.orientation import ORIENT_CKPT
    from scanner.infer import MODEL_PATH

    C.check_imported_from(scan, str(root))
    git = C.git_info(str(root))
    runner_sha256 = C.file_sha256(__file__)
    generator_sha256 = C.file_sha256(hw.__file__)
    torch.set_num_threads(threads)
    torch.manual_seed(C.SEED)
    cache = Path(os.getenv("DOCTR_CACHE_DIR", Path.home() / ".cache/doctr")) / "models"
    detector = Path(common.DET_CKPT)
    if not detector.is_file():
        detector = cache / "fast_base-688a8b34.pt"
    weights = [detector, Path(models.REC_CKPT), Path(ORIENT_CKPT), Path(scriptclf.CKPT), Path(MODEL_PATH)]
    weights.extend(sorted(Path(HW_DIR).glob("*.safetensors")))
    from ocr.api import weights_file_ok  # a file under 1 kB is a failed download, not weights
    if not weights_file_ok(str(detector)) or not any(p.suffix == ".safetensors" for p in weights):
        raise SystemExit("Full-page benchmark needs the local FAST detector and handwriting checkpoint: "
                         f"{common.DET_CKPT} or {cache / 'fast_base-688a8b34.pt'} (at least 1 kB), and "
                         "checkpoints/handwriting_v1/*.safetensors.")
    hashes = {p.name: C.file_sha256(p) for p in weights}
    scanner = scan.Scanner()
    # Per-request settings, set the way ocr/api.py read_page sets them on the shared Scanner.
    scanner.ocr.use_scanner = bool(page_scanner)
    scanner.do_deskew = bool(deskew)
    mods = dict(torch=torch, models=models, scriptclf=scriptclf, hwdata=hwdata, scan=scan,
                hw=scanner.hw, rec=scanner.ocr.rec, clf=scanner.clf, read_kw={})
    fonts = hw.font_files(str(root))
    font_sha256 = {Path(f).name: C.file_sha256(f) for f in fonts}
    trace, geometry = [], Counter()
    for seed in seeds:
        pg = hw.render_page(seed, fonts)
        image = Image.fromarray(pg["image"])
        start = time.perf_counter()
        result = scanner.run(image)
        elapsed = time.perf_counter() - start
        ref = [ln["text"] for ln in pg["lines"]]
        pred = [ln["text"] for ln in result["lines"]]
        turns = result["preprocessing"]["quarter_turns"]
        boxes = [bounds([result["words"][w]["source_quad"] for w in ln["words"]]) for ln in result["lines"]]
        layout = hw.match_lines(boxes, [rotated_bounds(ln["quad"], *image.size, turns) for ln in pg["lines"]])
        geometry.update(layout)
        annotated = hw.read_pages([pg], mods, {k: [] for k in ("printed_ms", "route_ms", "handwriting_ms")})
        print(f"Scored page {seed}", flush=True)
        trace.append(dict(seed=seed, image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                          reference=ref, detected=pred, detected_lines=result["lines"], source_boxes=boxes,
                          alignment=align(ref, pred), annotated=[ln["shipped"] for ln in annotated],
                          annotated_readers=annotated, layout=layout, preprocessing=result["preprocessing"],
                          scanner=result["scanner"], elapsed_s=elapsed))
    return dict(git=git, model_sha256=hashes, device="cpu", threads=threads,
                font_sha256=font_sha256,
                settings="Scanner() defaults, RGB, seed 2026", seeds=list(seeds),
                scanner_settings=scanner_settings(page_scanner, deskew),
                runner_sha256=runner_sha256, generator_sha256=generator_sha256,
                versions={p: importlib.metadata.version(p) for p in
                          ("torch", "torchvision", "python-doctr", "transformers", "numpy", "scipy", "pillow")},
                paired_line_outcomes=paired_outcomes(trace), layout=dict(geometry),
                scores={k: score(trace, k) for k in ("annotated", "detected")}, pages=trace)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["validation", "regression", "holdout"], default="regression")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--page-scanner", choices=["on", "off"], default="on",
                    help="page-scanner stage (default on, as Scanner() and earlier published runs; the app "
                         "sends scanner=0, i.e. off)")
    ap.add_argument("--deskew", choices=["on", "off"], default="off")
    a = ap.parse_args()
    if a.threads < 1 or (a.limit is not None and a.limit < 1):
        ap.error("threads and limit must be positive")
    splits = dict(validation=hw.VAL_SEEDS, regression=hw.TEST_SEEDS, holdout=range(4100, 4130))
    C.limit_threads(a.threads)
    report = run(a.root, splits[a.split][:a.limit], a.threads, a.page_scanner == "on", a.deskew == "on")
    suffix = "" if (a.page_scanner, a.deskew) == ("on", "off") else f"_scanner-{a.page_scanner}_deskew-{a.deskew}"
    print(C.write_result(a.out, C.out_name("full_pages_" + a.split + suffix, a.limit), report))

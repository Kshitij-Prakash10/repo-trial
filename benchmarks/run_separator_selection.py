"""Evaluate an experimental reader selector on annotated crops, without modifying either transcription.

Generate printed controls with --printed-out. Score validation and both controls with --holdout and
--printed. The holdout is the paired Scanner.run benchmark's JSON (4100..4129). The stored validation
reader outputs are fixed before the experiment. All inputs are synthetic and use seed 2026.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from time import perf_counter

from benchmarks import common as C, run_handwriting as hw

# A visible decimal point, decimal comma, or slash within a numeric token. Consider a single dropped
# separator or a slash read as 1 or 7. Never manufacture a separator or rewrite an OCR candidate.
SEPARATOR = re.compile(r"(?<=\d)[/.,](?=\d)")
VALIDATION = Path("reports/review_2026-11/after_ce56e17/handwriting.json")
FONT_ROOT = Path("/usr/share/fonts/truetype")
PRINT_FONTS = ("dejavu/DejaVuSans.ttf", "dejavu/DejaVuSerif.ttf", "dejavu/DejaVuSansMono.ttf",
               "liberation/LiberationSans-Regular.ttf", "liberation/LiberationSerif-Regular.ttf",
               "liberation/LiberationMono-Regular.ttf")


def select(shipped: str, printed: str, chosen_hw: bool, context_edits: int) -> str:
    if not chosen_hw or shipped == printed:
        return shipped
    p, h = printed.casefold().split(), shipped.casefold().split()
    if len(p) != len(h):
        return shipped
    for index, (word, other) in enumerate(zip(p, h)):
        if word == other or not re.fullmatch(r"[\d.,/]+", word):
            continue
        for mark in SEPARATOR.finditer(word):
            replacements = ("", "1", "7") if mark.group() == "/" else ("",)
            if any(word[:mark.start()] + c + word[mark.end():] == other for c in replacements):
                if sum(hw.edit(x, y) for j, (x, y) in enumerate(zip(p, h)) if j != index) <= context_edits:
                    return printed
    return shipped


def printed_controls(path: Path, threads: int):
    C.limit_threads(threads)
    import torch
    from ocr import hwdata, models, scan, scriptclf
    from ocr.handwriting import HandwritingReader

    torch.set_num_threads(threads)
    torch.manual_seed(C.SEED)
    mods = dict(torch=torch, hwdata=hwdata, models=models, scan=scan, scriptclf=scriptclf,
                hw=HandwritingReader(local_only=True), rec=models.rec_predictor(models.load_rec(ckpt=models.REC_CKPT)),
                clf=scriptclf.load_clf())
    fonts = [str(FONT_ROOT / f) for f in PRINT_FONTS]
    latency = {key: [] for key in ("printed_ms", "route_ms", "handwriting_ms")}
    rows = hw.read_pages((hw.render_page(i, fonts) for i in range(5100, 5120)), mods, latency)
    result = {"threads": threads, "fonts": {f: C.file_sha256(f) for f in fonts}, "rows": rows,
              "latency_ms": {k: sum(v) / len(v) for k, v in latency.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))


def measure(rows, context_edits):
    base, revised, changed = [], [], []
    start = perf_counter()
    for r in rows:
        value = select(r["shipped"], r["printed"], r["routed"] and r["shipped"] != r["printed"],
                       context_edits)
        if r["shipped"] != value:
            changed.append({"page": r["page"], "line": r["line"], "before": r["shipped"], "after": value,
                            "reference": r["truth"]})
        base.append((r["shipped"], r["truth"]))
        revised.append((value, r["truth"]))
    elapsed = (perf_counter() - start) * 1000 / len(rows)

    def scores(pairs):
        decimals = [(p, t) for p, t in pairs if any("." in x for x in hw.numbers(t))]
        return {**hw.cer_wer(pairs), "exact": sum(p == t for p, t in pairs),
                "fields": hw.field_scores(pairs), "numeric_line_exact": hw.numeric_line(pairs),
                "numeric_sequences_on_decimal_lines": C.frac(sum(hw.numbers(p) == hw.numbers(t) for p, t in decimals),
                                                             len(decimals))}
    return {"baseline": scores(base), "experiment": scores(revised), "changed": changed,
            "selector_ms_per_line": elapsed}


def run(holdout: Path, printed: Path, output: Path):
    stored = json.loads(VALIDATION.read_text())
    validation = stored["validation"]["rows"]
    candidates = {m: measure(validation, m) for m in (0, 1)}
    # Smallest validation CER, then largest number of exact lines, then the narrower selector.
    selected = min(candidates, key=lambda m: (candidates[m]["experiment"]["char_errors"],
                                              -candidates[m]["experiment"]["exact"], m))
    pages = json.loads(holdout.read_text())["pages"]
    if [p["seed"] for p in pages] != list(range(4100, 4130)):
        raise ValueError("Expected the complete independent handwriting holdout of 30 pages")
    control = json.loads(printed.read_text())
    if sorted({r["page"] for r in control["rows"]}) != list(range(5100, 5120)):
        raise ValueError("Expected 20 independent printed control pages")
    result = {"validation_source_sha256": C.file_sha256(VALIDATION), "paired_holdout_sha256": C.file_sha256(holdout),
              "printed_control_sha256": C.file_sha256(printed), "validation": candidates,
              "chosen_context_edits": selected,
              "fixed_old_regression": measure(stored["test"]["rows"], selected),
              "handwriting_holdout": measure([r for p in pages for r in p["annotated_readers"]], selected),
              "printed_holdout": measure(control["rows"], selected)}
    # Apply the frozen decision to the detector's actual routed/selected lines, then re-score page text.
    pairs, old_pairs, changes = [], [], []
    for p in pages:
        preds = []
        for j, line in enumerate(p["detected_lines"]):
            chosen = line["text"]
            revised = select(chosen, line.get("printed_text", chosen), line["script"] == "handwritten", selected)
            preds.append(revised)
            if chosen != revised:
                changes.append({"seed": p["seed"], "line": j, "before": chosen, "after": revised})
        ref = hw.norm(" ".join(p["reference"]))
        old_pairs.append((hw.norm(" ".join(p["detected"])), ref))
        pairs.append((hw.norm(" ".join(preds)), ref))
    result["full_page_holdout"] = {"baseline": hw.cer_wer(old_pairs), "experiment": hw.cer_wer(pairs),
                                   "changed_lines": changes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--printed-out", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--printed", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.printed_out:
        printed_controls(args.printed_out, args.threads)
    elif args.holdout and args.printed and args.out:
        run(args.holdout, args.printed, args.out)
    else:
        parser.error("Specify --printed-out or all of --holdout, --printed, and --out.")

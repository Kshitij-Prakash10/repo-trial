"""Evaluate an OCR system on the test split of every dataset: recognition, detection, end to end.

    python -m ocr.evaluate --system pretrained      # docTR weights, no fine-tuning
    python -m ocr.evaluate --system finetuned       # models/ocr/*.pt
    python -m ocr.evaluate --system tesseract       # Tesseract 5 (eng)

Metrics (test split only; --split val exists for development):
  recognition  CER / WER on word crops cut with the ground-truth boxes (SROIE: words derived from its
               line boxes, see ocr/data.py). Corpus level: total edits / total reference length.
  detection    precision / recall / F1 of word boxes at IoU 0.5, detector on the raw image.
  end to end   CER / WER of the full-page text, raw image -> final text (scanner stage included).
               Reference and prediction are both put in reading order by ocr.common.group_lines, so
               the metric scores what was read, not how a layout was linearised.
Outputs: runs/eval/<system>/<dataset>_<task>.jsonl (per item) and runs/eval/<system>_<split>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from ocr.common import DATASETS, ROOT, error_counts, load_pages, match_boxes, page_text, prf, rates
from ocr.data import usable, word_crops, word_pages

EVAL_DIR = os.path.join(ROOT, "runs", "eval")
TESS_H = 48  # px; word crops are scaled to this height and given a white border for Tesseract


def _tess_words(crops: list[np.ndarray]) -> list[str]:
    """Tesseract on a batch of word crops in one process (one image per "page", --psm 8: single word).

    Each crop is scaled to TESS_H px high and given a border in its own background colour, which is
    how Tesseract expects a word image. One thread per process (OMP_THREAD_LIMIT=1)."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        names = []
        for i, crop in enumerate(crops):
            h, w = crop.shape[:2]
            s = TESS_H / h
            im = Image.fromarray(crop).resize((max(8, int(round(w * s))), TESS_H), Image.BICUBIC)
            edge = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
            canvas = Image.new("RGB", (im.width + 24, im.height + 24), tuple(int(v) for v in np.median(edge, axis=0)))
            canvas.paste(im, (12, 12))
            names.append(os.path.join(d, f"{i:06d}.png"))
            canvas.save(names[-1])
        with open(os.path.join(d, "list.txt"), "w") as f:
            f.write("\n".join(names))
        r = subprocess.run(["tesseract", os.path.join(d, "list.txt"), "stdout", "-l", "eng", "--psm", "8", "--oem", "1",
                            "-c", "page_separator=\x0c"], capture_output=True, text=True,
                           env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    out = [t.strip() for t in r.stdout.split("\x0c")]
    if len(out) == len(crops) + 1 and not out[-1]:
        out = out[:-1]
    if len(out) != len(crops):
        raise RuntimeError(f"tesseract returned {len(out)} results for {len(crops)} crops: {r.stderr[-300:]}")
    return out


def _tess_page(path: str) -> dict:
    import pytesseract
    os.environ["OMP_THREAD_LIMIT"] = "1"  # one thread per worker process
    with Image.open(path) as im:
        d = pytesseract.image_to_data(im.convert("RGB"), lang="eng", config="--psm 3 --oem 1",
                                      output_type=pytesseract.Output.DICT)
    words = [(t.strip(), [x, y, x + w, y + h]) for t, x, y, w, h, lvl in
             zip(d["text"], d["left"], d["top"], d["width"], d["height"], d["level"]) if lvl == 5 and t.strip()]
    return {"texts": [w[0] for w in words], "boxes": [w[1] for w in words]}


def _write(system: str, name: str, rows: list[dict]) -> None:
    d = os.path.join(EVAL_DIR, system)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def gt_boxes(page: dict) -> np.ndarray:
    return np.array([it["box"] for it in page["items"] if usable(it)], float).reshape(-1, 4)


def reference_text(page: dict) -> str:
    """Full-page reference text from the dataset's own items (words, or lines for SROIE), in reading order."""
    items = [it for it in page["items"] if it["text"].strip()]
    return page_text([it["text"] for it in items], np.array([it["box"] for it in items], float).reshape(-1, 4))


def eval_recognition(system: str, split: str, limit: int | None, workers: int) -> dict:
    out = {}
    rec = None
    if system != "tesseract":
        from ocr import models
        rec = models.rec_predictor(models.load_rec(ckpt=models.REC_CKPT if system == "finetuned" else None))
    for ds in DATASETS:
        data = list(word_crops(ds, split, limit))
        crops, texts, ids = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
        t0 = time.time()
        if rec is None:
            chunks = [crops[i:i + 200] for i in range(0, len(crops), 200)]
            with Pool(workers) as pool:
                preds = [t for part in pool.map(_tess_words, chunks) for t in part]
        else:
            preds = [p[0] for p in rec(crops)]
        counts = [error_counts(p, t, ds) for p, t in zip(preds, texts)]
        out[ds] = {**rates(counts), "seconds": round(time.time() - t0)}
        _write(f"{system}_{split}", f"{ds}_rec", [{"page": i, "index": k, "true": t, "pred": p, **c}
                                                  for k, (i, t, p, c) in enumerate(zip(ids, texts, preds, counts))])
        print(system, "recognition", ds, json.dumps(out[ds]), flush=True)
    return out


def eval_pages(system: str, split: str, limit: int | None, workers: int) -> tuple[dict, dict]:
    """Detection and end-to-end results in one pass over the pages."""
    det_out, e2e_out = {}, {}
    pipe = None
    if system != "tesseract":
        from ocr.pipeline import OcrPipeline
        pipe = OcrPipeline.load(system)
    for ds in DATASETS:
        pages = load_pages(ds, split)[:limit]
        wpages = word_pages(ds, split)[:limit]  # word-level boxes for the detection score
        t0 = time.time()
        if pipe is None:
            with Pool(workers) as pool:
                results = pool.map(_tess_page, [p["path"] for p in pages], chunksize=2)
        else:
            results = []
            for p in pages:
                with Image.open(p["path"]) as im:
                    r = pipe.run(im, keep_first_pass=True)
                row = {"texts": [w["text"] for w in r["words"]], "text": r["text"], "boxes": r["first_pass_boxes"],
                       "scanner_used": r["scanner"]["used"], "scanner_reason": r["scanner"]["reason"]}
                if r["scanner"]["used"]:  # the same page with the scanner stage off, for the ablation
                    pipe.use_scanner = False
                    with Image.open(p["path"]) as im:
                        row["text_no_scanner"] = pipe.run(im)["text"]
                    pipe.use_scanner = True
                results.append(row)
        secs = time.time() - t0
        tp = n_pred = n_true = 0
        counts, counts_ns, rows_det, rows_e2e, used = [], [], [], [], 0
        for p, wp, r in zip(pages, wpages, results):
            pb, tb = np.array(r["boxes"], float).reshape(-1, 4), gt_boxes(wp)
            m = len(match_boxes(pb, tb))
            tp, n_pred, n_true = tp + m, n_pred + len(pb), n_true + len(tb)
            rows_det.append({"page": p["id"], "tp": m, "n_pred": len(pb), "n_true": len(tb)})
            ref = reference_text(p)
            text = r["text"] if "text" in r else page_text(r["texts"], pb)
            c = error_counts(text, ref, ds)
            counts.append(c)
            counts_ns.append(error_counts(r["text_no_scanner"], ref, ds) if "text_no_scanner" in r else c)
            used += bool(r.get("scanner_used"))
            rows_e2e.append({"page": p["id"], "true": ref, "pred": text, **c,
                             "scanner_used": r.get("scanner_used"), "scanner_reason": r.get("scanner_reason")})
        det_out[ds] = prf(tp, n_pred, n_true)
        e2e_out[ds] = {**rates(counts), "pages": len(pages), "seconds_per_page": round(secs / max(len(pages), 1), 2)}
        if pipe is not None:
            e2e_out[ds]["scanner_crop_used_on_pages"] = used
            e2e_out[ds]["without_scanner_stage"] = {k: v for k, v in rates(counts_ns).items() if k in ("cer", "wer")}
        _write(f"{system}_{split}", f"{ds}_det", rows_det)
        _write(f"{system}_{split}", f"{ds}_e2e", rows_e2e)
        print(system, "detection", ds, json.dumps(det_out[ds]), flush=True)
        print(system, "end_to_end", ds, json.dumps(e2e_out[ds]), flush=True)
    return det_out, e2e_out


def main(system: str, split: str = "test", limit: int | None = None, workers: int = 4) -> dict:
    import torch
    torch.set_num_threads(workers)
    out = {"system": system, "split": split, "limit_pages": limit}
    out["recognition"] = eval_recognition(system, split, limit, workers)
    out["detection"], out["end_to_end"] = eval_pages(system, split, limit, workers)
    os.makedirs(EVAL_DIR, exist_ok=True)
    name = f"{system}_{split}" + (f"_first{limit}" if limit else "")
    with open(os.path.join(EVAL_DIR, name + ".json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", required=True, choices=["pretrained", "finetuned", "tesseract"])
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--limit", type=int, default=None, help="first N pages per dataset (smoke tests only)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    main(a.system, a.split, a.limit, a.workers)

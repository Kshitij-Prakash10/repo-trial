"""Evaluate the upgraded scanner (handwriting + orientation) on the test splits.

    python -m ocr.evaluate_v2 --task printed        # FUNSD / SROIE / CORD end to end, vs reports/baseline.json
    python -m ocr.evaluate_v2 --task iam            # IAM test lines (handwriting recogniser alone)
    python -m ocr.evaluate_v2 --task gnhk           # GNHK test: lines (recogniser alone) and pages end to end
    python -m ocr.evaluate_v2 --task orientation    # 0/90/180/270 accuracy on all test pages, + Tesseract OSD
    python -m ocr.evaluate_v2 --task rotated        # end-to-end CER of rotated pages vs upright
    python -m ocr.evaluate_v2 --task all

Test splits only (--split val exists for development). CER / WER are corpus level, as in
ocr/evaluate.py; SROIE is scored without case. Per-item rows go to runs/eval_v2/<task>/, summaries to
reports/eval_v2/<task>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from PIL import Image

from ocr import hwdata
from ocr.common import DATASETS, ROOT, error_counts, load_pages, page_text, rates
from ocr.evaluate import reference_text

RUNS = os.path.join(ROOT, "runs", "eval_v2")
REPORTS = os.path.join(ROOT, "reports", "eval_v2")
SEED = 2026


def _save(task: str, split: str, summary: dict, rows: dict[str, list]) -> None:
    os.makedirs(os.path.join(RUNS, task), exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    for name, r in rows.items():
        with open(os.path.join(RUNS, task, f"{name}_{split}.jsonl"), "w") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in r)
    with open(os.path.join(REPORTS, f"{task}{'' if split == 'test' else '_' + split}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(task, json.dumps(summary)[:1500], flush=True)


def _scripts(r: dict) -> dict:
    n = len(r["lines"])
    return {"lines": n, "handwritten_lines": sum(ln["script"] == "handwritten" for ln in r["lines"]),
            "low_confidence_lines": sum(ln["low_confidence"] for ln in r["lines"])}


def gnhk_reference(page: dict) -> str:
    """Page reference text: every annotated word except GNHK's special marks, in reading order."""
    ws = [w for w in page["words"] if w["text"] not in hwdata.GNHK_SPECIAL and w["text"].strip()]
    boxes = np.array([[min(p[0] for p in w["poly"]), min(p[1] for p in w["poly"]),
                       max(p[0] for p in w["poly"]), max(p[1] for p in w["poly"])] for w in ws], float).reshape(-1, 4)
    return page_text([w["text"] for w in ws], boxes)


def task_printed(split: str, limit: int | None) -> None:
    from ocr.scan import Scanner
    sc = Scanner()
    summary, rows = {"what": "printed test sets, full upgraded pipeline, end to end"}, {}
    with open(os.path.join(ROOT, "reports", "baseline.json")) as f:
        base = json.load(f)["end_to_end"]
    for ds in DATASETS:
        counts, r_rows, t0, hw, lines, turned, skewed = [], [], time.time(), 0, 0, 0, 0
        pages = load_pages(ds, split)[:limit]
        for p in pages:
            with Image.open(p["path"]) as im:
                r = sc.run(im)
            c = error_counts(r["text"], reference_text(p), ds)
            counts.append(c)
            s = _scripts(r)
            hw, lines = hw + s["handwritten_lines"], lines + s["lines"]
            turned += r["preprocessing"]["quarter_turns"] != 0
            skewed += r["preprocessing"]["deskew_degrees"] != 0
            r_rows.append({"page": p["id"], "pred": r["text"], **c, **s, **r["preprocessing"]})
        res = rates(counts)
        summary[ds] = {"cer": res["cer"], "wer": res["wer"], "pages": len(pages),
                       "seconds_per_page": round((time.time() - t0) / max(len(pages), 1), 2),
                       "lines_routed_to_handwriting": hw, "lines": lines, "pages_turned": turned, "pages_deskewed": skewed,
                       "baseline_cer": base[ds]["cer"], "baseline_wer": base[ds]["wer"],
                       "cer_change_points": round(100 * (res["cer"] - base[ds]["cer"]), 2) if limit is None and split == "test" else None}
        rows[ds] = r_rows
        print(ds, summary[ds], flush=True)
    _save("printed", split, summary, rows)


def task_iam(split: str, limit: int | None) -> None:
    from ocr.handwriting import HandwritingReader
    from ocr.postcorrect import correct_line
    summary, rows = {"what": "IAM lines, official writer-independent split"}, {}
    for name, path in (("pretrained", "microsoft/trocr-small-handwritten"), ("finetuned", None)):
        reader = HandwritingReader(path)
        items = list(hwdata.iam_lines("validation" if split == "val" else split, limit))
        t0 = time.time()
        preds = reader.read([np.asarray(im) for _, im, _ in items])
        secs = time.time() - t0
        counts = [error_counts(p[0], t) for p, (_, _, t) in zip(preds, items)]
        fixed = [error_counts(correct_line(p[0]), t) for p, (_, _, t) in zip(preds, items)]
        summary[name] = {**rates(counts), "seconds_per_line": round(secs / max(len(items), 1), 3), "model": reader.name,
                         "with_lexicon_correction": {k: rates(fixed)[k] for k in ("cer", "wer")}}
        rows[f"iam_{name}"] = [{"id": i, "true": t, "pred": p[0], "confidence": round(p[1], 4), **c}
                               for (i, _, t), p, c in zip(items, preds, counts)]
    _save("iam", split, summary, rows)


def task_gnhk(split: str, limit: int | None) -> None:
    from ocr.handwriting import HandwritingReader
    from ocr.postcorrect import correct_line
    from ocr.scan import Scanner
    pages = hwdata.gnhk_pages(split)[:limit]
    summary, rows = {"what": "GNHK (photographed handwriting), official test pages"}, {}
    lines = [(p["id"], ln) for p in pages for ln in hwdata.gnhk_lines(p) if not ln["special"] and ln["text"].strip()]
    for name, path in (("pretrained", "microsoft/trocr-small-handwritten"), ("finetuned", None)):
        reader = HandwritingReader(path)
        preds = reader.read([ln["crop"] for _, ln in lines])
        counts = [error_counts(p[0], ln["text"]) for p, (_, ln) in zip(preds, lines)]
        hw = [c for c, (_, ln) in zip(counts, lines) if ln["type"] == "H"]
        summary[f"lines_{name}"] = {"all": rates(counts), "handwritten_only": rates(hw),
                                    "with_lexicon_correction": {k: rates([error_counts(correct_line(p[0]), ln["text"])
                                                                          for p, (_, ln) in zip(preds, lines)])[k] for k in ("cer", "wer")}}
        rows[f"gnhk_lines_{name}"] = [{"page": i, "line": ln["line"], "type": ln["type"], "true": ln["text"], "pred": p[0],
                                       "confidence": round(p[1], 4), **c} for (i, ln), p, c in zip(lines, preds, counts)]
    del reader
    for name, kw in (("full", {}), ("printed_reader_only", {"handwriting": False})):
        sc = Scanner(**kw)
        counts, fixed, r_rows, t0, hw, n_lines = [], [], [], time.time(), 0, 0
        for p in pages:
            r = sc.run(Image.fromarray(hwdata.load_image(p["path"])))
            ref = gnhk_reference(p)
            c = error_counts(r["text"], ref)
            counts.append(c)
            fixed.append(error_counts("\n".join(correct_line(ln["text"]) if ln["script"] == "handwritten" else ln["text"]
                                                for ln in r["lines"]), ref))
            s = _scripts(r)
            hw, n_lines = hw + s["handwritten_lines"], n_lines + s["lines"]
            r_rows.append({"page": p["id"], "true": ref, "pred": r["text"], **c, **s, **r["preprocessing"]})
        summary[f"pages_{name}"] = {**rates(counts), "pages": len(pages), "seconds_per_page": round((time.time() - t0) / max(len(pages), 1), 2),
                                    "lines": n_lines, "lines_routed_to_handwriting": hw,
                                    "with_lexicon_correction": {k: rates(fixed)[k] for k in ("cer", "wer")}}
        rows[f"gnhk_pages_{name}"] = r_rows
        print(name, summary[f"pages_{name}"], flush=True)
    _save("gnhk", split, summary, rows)


def _all_pages(split: str, limit: int | None) -> list[tuple[str, str]]:
    out = [(ds, p["path"]) for ds in DATASETS for p in load_pages(ds, split)[:limit]]
    return out + [("gnhk", p["path"]) for p in hwdata.gnhk_pages(split)[:limit]]


def task_orientation(split: str, limit: int | None, osd_pages: int = 25) -> None:
    import pytesseract
    from ocr import orientation as O
    summary, rows = {"what": "orientation accuracy, every test page at 0 / 90 / 180 / 270 degrees"}, {"orientation": []}
    rng = np.random.default_rng(SEED)
    for name, ckpt in (("pretrained_doctr", None), ("finetuned", O.ORIENT_CKPT)):
        model = O.load_orient(ckpt)
        hit: dict[str, list[int]] = {}
        for ds, path in _all_pages(split, limit):
            img = hwdata.load_image(path)
            img = np.asarray(Image.fromarray(img).resize((max(1, img.shape[1] * 1024 // max(img.shape[:2])),
                                                         max(1, img.shape[0] * 1024 // max(img.shape[:2])))))
            for k in range(4):
                pk, conf = O.predict_k(model, np.ascontiguousarray(np.rot90(img, k)))
                hit.setdefault(ds, []).append(int(pk == k))
                if name == "finetuned":
                    rows["orientation"].append({"dataset": ds, "page": os.path.basename(path), "k": k, "pred": pk, "confidence": round(conf, 4)})
        summary[name] = {**{ds: round(float(np.mean(v)), 4) for ds, v in hit.items()},
                         "all": round(float(np.mean([x for v in hit.values() for x in v])), 4), "n": sum(len(v) for v in hit.values())}
    # Tesseract OSD on a fixed random subset (it is slow and fails on pages with little text)
    hit, failed = {}, 0
    for ds in list(DATASETS) + ["gnhk"]:
        paths = [p for d, p in _all_pages(split, limit) if d == ds]
        for i in rng.permutation(len(paths))[:osd_pages]:
            img = hwdata.load_image(paths[int(i)])
            for k in range(4):
                try:
                    osd = pytesseract.image_to_osd(Image.fromarray(np.ascontiguousarray(np.rot90(img, k))), output_type=pytesseract.Output.DICT)
                    pk = (osd["rotate"] // 90) % 4  # degrees clockwise needed to fix = quarter-turns CCW applied
                except Exception:
                    pk, failed = -1, failed + 1
                hit.setdefault(ds, []).append(int(pk == k))
    summary["tesseract_osd"] = {**{ds: round(float(np.mean(v)), 4) for ds, v in hit.items()},
                                "all": round(float(np.mean([x for v in hit.values() for x in v])), 4),
                                "n": sum(len(v) for v in hit.values()), "failed_to_answer": failed,
                                "note": f"{osd_pages} random pages per dataset (seed {SEED}); a failure counts as wrong"}
    _save("orientation", split, summary, rows)


def task_rotated(split: str, limit: int | None, per_set: int = 25) -> None:
    from ocr import orientation as O
    from ocr.scan import Scanner
    sc = Scanner()
    rng = np.random.default_rng(SEED)
    summary, rows = {"what": f"end-to-end CER of the same pages at 0 / 90 / 180 / 270 degrees ({per_set} random test pages per set, seed {SEED}), "
                             "and with a small skew of up to 10 degrees"}, {"rotated": []}
    for ds in list(DATASETS) + ["gnhk"]:
        pages = hwdata.gnhk_pages(split) if ds == "gnhk" else load_pages(ds, split)
        pick = [pages[int(i)] for i in rng.permutation(len(pages))[:(limit or per_set)]]
        counts = {k: [] for k in ("0", "90", "180", "270", "skewed")}
        for p in pick:
            img = hwdata.load_image(p["path"])
            ref = gnhk_reference(p) if ds == "gnhk" else reference_text(p)
            variants = {str(90 * k): np.ascontiguousarray(np.rot90(img, k)) for k in range(4)}
            variants["skewed"] = O.rotate(img, float(rng.uniform(3, 10) * rng.choice([-1, 1])))
            for name, v in variants.items():
                r = sc.run(Image.fromarray(v))
                c = error_counts(r["text"], ref, ds)
                counts[name].append(c)
                rows["rotated"].append({"dataset": ds, "page": p["id"], "variant": name, **c, **r["preprocessing"]})
        summary[ds] = {name: rates(c)["cer"] for name, c in counts.items()}
        summary[ds]["worst_minus_upright_points"] = round(100 * (max(summary[ds][a] for a in ("90", "180", "270")) - summary[ds]["0"]), 2)
        summary[ds]["pages"] = len(pick)
        print(ds, summary[ds], flush=True)
    _save("rotated", split, summary, rows)


TASKS = {"printed": task_printed, "iam": task_iam, "gnhk": task_gnhk, "orientation": task_orientation, "rotated": task_rotated}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, choices=list(TASKS) + ["all"])
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--limit", type=int, default=None, help="first N pages / lines per set (smoke tests only)")
    a = ap.parse_args()
    for t in (TASKS if a.task == "all" else [a.task]):
        TASKS[t](a.split, a.limit)

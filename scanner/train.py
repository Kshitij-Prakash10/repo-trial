"""Train and evaluate the page segmenter. Data: the stored SYNTHETIC set in data/synthetic/.

Protocol: train, val, test and ood photos come from disjoint generator seeds, so no evaluation photo is
seen in training. The model is a per-cell HistGradientBoostingClassifier (stage 1) plus an auto-context
stage 2 that also sees smoothed stage-1 probabilities (trained on out-of-fold stage-1 maps). The
post-processing (component, quad fit, edge refinement) is shared with the non-ML baseline (Otsu threshold
on brightness), so the comparison isolates the page mask. The "ood" split uses a generator setting never
used in training (other layouts, chequered and granite backgrounds, stronger tilt, warm light).

The functions below take generic items {"image": PIL image or path, "corners": (4, 2) px TL TR BR BL,
"meta": dict} so that real photos (for example a later fine-tune) can use the same code:
image_rows / build_rows -> fit -> evaluate -> report.

CLI: python -m scanner.train   (uses data/synthetic/, generating it first when it is missing)
"""
from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from multiprocessing import get_context

import joblib
import numpy as np
import sklearn
from PIL import Image, ImageDraw
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from scanner import dataset
from scanner import features as feats
from scanner import geometry as geo
from scanner import infer
from scanner import synth

ROOT = infer.ROOT
MODEL_PATH = infer.MODEL_PATH
METRICS_PATH = infer.METRICS_PATH
EXAMPLES_DIR = os.path.join(ROOT, "data", "scanner_examples")
MODEL_VERSION = "scanner-0.1"
SEED = 11
CELLS_PER_IMAGE = 3000  # grid cells sampled per training photo
WORKERS = min(4, os.cpu_count() or 1)


def true_mask(corners: np.ndarray, W: int, H: int) -> np.ndarray:
    """(H, W) bool page mask from the 4 true corners (photo px)."""
    return synth.poly_mask(corners, W, H) >= 0.5


def image_rows(image: Image.Image, corners: np.ndarray, sample_seed: int,
               cells: int = CELLS_PER_IMAGE) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Features (float16, all cells), labels (page = cell at least half covered) and a seeded subsample index."""
    X, (gw, gh) = feats.compute(image)
    m = Image.fromarray(true_mask(corners, *image.size).astype(np.uint8) * 255).resize((gw, gh), Image.BOX)
    y = (np.asarray(m, np.float32).ravel() >= 127.5).astype(np.int8)
    rng = np.random.default_rng([SEED, sample_seed])
    idx = np.sort(rng.choice(len(y), size=min(cells, len(y)), replace=False))
    return X.astype(np.float16), y, idx, (gw, gh)


def _rows_one(args: tuple[dict, int]):
    item, i = args
    return image_rows(dataset.open_image(item), item["corners"], i)


def build_rows(items: list[dict], workers: int = WORKERS) -> list:
    """image_rows for every item (in parallel); the subsample seed is the item's position."""
    with get_context("spawn").Pool(workers) as pool:  # spawn: fork after OpenMP can deadlock
        return pool.map(_rows_one, [(it, i) for i, it in enumerate(items)], chunksize=8)


def _hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)


def fit(rows: list, stage2: bool = True, data_label: str = synth.DATA_LABEL) -> dict:
    """Fit stage 1 on all rows; stage 2 on out-of-fold stage-1 maps (2 folds by image). Returns the bundle."""
    X1 = np.concatenate([r[0][r[2]] for r in rows]).astype(np.float32)
    y = np.concatenate([r[1][r[2]] for r in rows])
    s1 = _hgb(SEED).fit(X1, y)
    s2 = None
    if stage2:
        fold = np.arange(len(rows)) % 2
        ctx = [None] * len(rows)
        for k in (0, 1):
            tr = [r for r, f in zip(rows, fold) if f != k]
            m = _hgb(SEED + 1 + k).fit(np.concatenate([r[0][r[2]] for r in tr]).astype(np.float32),
                                       np.concatenate([r[1][r[2]] for r in tr]))
            for i in np.nonzero(fold == k)[0]:
                X, _, idx, (gw, gh) = rows[i]
                p = m.predict_proba(X.astype(np.float32))[:, 1].reshape(gh, gw)
                ctx[i] = feats.context(p)[idx]
        s2 = _hgb(SEED + 3).fit(np.hstack([X1, np.concatenate(ctx)]), y)
    return {"stage1": s1, "stage2": s2, "grid": feats.GRID, "feature_names": feats.feature_names(),
            "model_version": MODEL_VERSION, "refine": True, "data": data_label}


_MODEL: dict | None = None
_ABLATIONS = False


def _init(model: dict, ablations: bool = False) -> None:
    global _MODEL, _ABLATIONS
    _MODEL, _ABLATIONS = model, ablations
    threadpool_limits(1)  # one OpenMP thread per worker process; more oversubscribes the CPU badly


def iou(q: np.ndarray, mask: np.ndarray) -> float:
    """IoU of quad q (photo px, clipped to the frame) with a (H, W) bool mask."""
    H, W = mask.shape
    pm = synth.poly_mask(q, W, H) >= 0.5
    u = (pm | mask).sum()
    return float((pm & mask).sum() / u) if u else 0.0


def evaluate_one(model: dict, item: dict, ablations: bool = False) -> dict:
    """Model and baseline (and optional ablations) on one item: IoU, corner error, page_found, confidence."""
    img = dataset.open_image(item)
    mask = true_mask(item["corners"], *img.size)
    out = {"meta": item.get("meta", {})}
    variants = {"ml": lambda: infer.detect(img, model),
                "baseline": lambda: infer.detect_from_prob(img, infer.otsu_prob(img, model["grid"]))}
    if ablations and model.get("stage2") is not None:
        variants["ml_stage1_only"] = lambda: infer.detect(img, dict(model, stage2=None))
        variants["ml_no_refine"] = lambda: infer.detect(img, dict(model, refine=False))
    for name, fn in variants.items():
        r = fn()
        q = r["corners"]
        out[name] = {"iou": iou(q, mask), "corner_err_pct": geo.corner_error(q, item["corners"]),
                     "page_found": bool(r["page_found"]), "confidence": float(r["confidence"])}
    return out


def _eval_worker(item: dict) -> dict:
    return evaluate_one(_MODEL, item, _ABLATIONS)


def evaluate(model: dict, items: list[dict], ablations: bool = False, workers: int = WORKERS) -> list[dict]:
    """evaluate_one for every item, in parallel."""
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(model, ablations)) as pool:
        return pool.map(_eval_worker, items, chunksize=4)


def _summary(rows: list[dict], key: str) -> dict:
    iou_ = np.array([r[key]["iou"] for r in rows])
    err = np.array([r[key]["corner_err_pct"] for r in rows])
    found = np.array([r[key]["page_found"] for r in rows])
    return {"n_images": len(rows), "mask_iou_mean": round(float(iou_.mean()), 4),
            "mask_iou_median": round(float(np.median(iou_)), 4),
            "corner_err_pct_diag_mean": round(float(err.mean()), 3),
            "corner_err_pct_diag_median": round(float(np.median(err)), 3),
            "corner_err_pct_diag_p90": round(float(np.percentile(err, 90)), 3),
            "success_rate_iou_ge_0.9": round(float((iou_ >= 0.9).mean()), 4),
            "page_found_rate": round(float(found.mean()), 4)}


def report(rows: list[dict]) -> dict:
    """Summary per method, per background, on pages cut by the frame, and a confidence check."""
    keys = [k for k in ("ml", "baseline", "ml_stage1_only", "ml_no_refine") if k in rows[0]]
    rep = {k: _summary(rows, k) for k in keys}
    bgs = sorted({r["meta"].get("background") for r in rows if r["meta"].get("background")})
    if bgs:
        rep["by_background"] = {}
        for bg in bgs:
            sub = [r for r in rows if r["meta"].get("background") == bg]
            rep["by_background"][bg] = {k: {"n": len(sub), **{m: _summary(sub, k)[m] for m in (
                "success_rate_iou_ge_0.9", "corner_err_pct_diag_median")}} for k in ("ml", "baseline")}
    cut = [r for r in rows if r["meta"].get("cut")]
    if cut:
        rep["page_cut_by_frame"] = {k: _summary(cut, k) for k in ("ml", "baseline")}
    conf = np.array([r["ml"]["confidence"] for r in rows])
    ok = np.array([r["ml"]["iou"] >= 0.9 for r in rows])
    hi = conf >= infer.MIN_CONFIDENCE
    rep["ml_confidence_check"] = {
        "threshold": infer.MIN_CONFIDENCE, "n_at_or_above": int(hi.sum()),
        "success_rate_at_or_above": round(float(ok[hi].mean()), 4) if hi.any() else None,
        "success_rate_below": round(float(ok[~hi].mean()), 4) if (~hi).any() else None}
    return rep


def time_scan(items: list[dict], n: int = 25) -> dict:
    """Wall time of the full scan() (detect, warp, enhance gray) per photo, one process, model loaded."""
    infer.load_model()
    ts = []
    for it in items[:n]:
        img = dataset.open_image(it)
        t = time.perf_counter()
        infer.scan(img, "gray")
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    return {"n_images": len(ts), "mean_ms": round(float(ts.mean()), 1),
            "p90_ms": round(float(np.percentile(ts, 90)), 1), "max_ms": round(float(ts.max()), 1),
            "photo_long_side_px": "480-1280", "machine": platform.machine(), "cpu_count": os.cpu_count()}


def save_examples(items: list[dict], prefix: str) -> list[str]:
    """Small PNGs: photo with true (green) and predicted (red) outline, and the rectified gray page."""
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    paths = []
    for it in items:
        img = dataset.open_image(it)
        r = infer.scan(img, "gray")
        im = img.copy()
        d = ImageDraw.Draw(im)
        w = max(2, im.size[0] // 200)
        d.polygon([tuple(p) for p in it["corners"]], outline=(0, 200, 0), width=w)
        d.polygon([tuple(p) for p in r["corners"]], outline=(230, 0, 0), width=w)
        im.thumbnail((360, 360))
        page = r["image"].convert("RGB")
        page.thumbnail((360, 360))
        canvas = Image.new("RGB", (im.size[0] + page.size[0] + 10, max(im.size[1], page.size[1])), (255, 255, 255))
        canvas.paste(im, (0, 0))
        canvas.paste(page, (im.size[0] + 10, 0))
        p = os.path.join(EXAMPLES_DIR, f"{prefix}_{it['meta'].get('seed', len(paths))}.png")
        canvas.quantize(64).save(p, optimize=True)
        paths.append(os.path.relpath(p, ROOT))
    return paths


def main(data_dir: str = dataset.DATA_DIR, examples: bool = True) -> dict:
    """Train on data/synthetic/train, evaluate on val, test and ood, save the model and metrics."""
    t0 = time.time()
    generated = False
    if not dataset.exists(data_dir):
        print(f"{data_dir} missing: generating it", flush=True)
        dataset.generate(data_dir)
        generated = True
    splits = {s: dataset.load_split(s, data_dir) for s in dataset.SPLITS}
    t_gen = time.time() - t0
    rows = build_rows(splits["train"])
    t_feat = time.time() - t0 - t_gen
    print(f"features: {len(rows)} training photos in {t_feat:.1f} s", flush=True)
    t1 = time.time()
    model = fit(rows)
    t_fit = time.time() - t1
    print(f"fit: {t_fit:.1f} s", flush=True)
    del rows
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    infer._load.cache_clear()
    res = {"val": evaluate(model, splits["val"], ablations=True),
           "test": evaluate(model, splits["test"], ablations=True),
           "ood": evaluate(model, splits["ood"])}
    seeds = {s: [r.start, r.stop - 1] for s, (r, _) in dataset.SPLITS.items()}
    metrics = {
        "model_version": MODEL_VERSION,
        "trained_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": "synthetic: rendered health documents composited onto rendered backgrounds (scanner.synth "
                f"{dataset.GENERATOR_VERSION}), stored in data/synthetic/; no real photos",
        "model": "per-cell HistGradientBoostingClassifier (200 iterations, 31 leaves) on a grid with long side "
                 f"{feats.GRID}, plus an auto-context stage 2 on out-of-fold stage-1 maps; then largest component, "
                 "quad fit and edge refinement",
        "baseline": "Otsu threshold on grid brightness, then the same component, quad fit and edge refinement",
        "seed": SEED, "generator_seeds": seeds,
        "n_train_images": len(splits["train"]), "n_train_cells": len(splits["train"]) * CELLS_PER_IMAGE,
        "features": feats.feature_names(),
        "metric_definitions": {
            "mask_iou": "IoU of the predicted quad (clipped to the frame) with the true page mask, full resolution",
            "corner_err_pct_diag": "mean distance of the 4 corners to the true corners, percent of the true page "
                                   "diagonal, best cyclic correspondence; a fallback counts with the image corners",
            "success": "mask IoU >= 0.9",
            "ml_stage1_only / ml_no_refine": "ablations: without the auto-context stage, without edge refinement"},
        "test": report(res["test"]),
        "val": report(res["val"]),
        "ood_test": dict(report(res["ood"]), note="generator style 'ood': page layouts, chequered and granite "
                                                  "backgrounds, tilt and light never used in training; still synthetic"),
        "train_time_s": {"generate_data": round(t_gen, 1) if generated else 0.0, "features": round(t_feat, 1),
                         "fit": round(t_fit, 1)},
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "python": platform.python_version(),
    }
    metrics["inference_time_scan_gray"] = time_scan(splits["test"])
    if examples:
        pick = {it["meta"]["seed"]: it for it in splits["test"] + splits["ood"]}
        metrics["examples"] = save_examples([pick[s] for s in (200_001, 200_004, 200_007, 200_010)], "test") + \
            save_examples([pick[s] for s in (300_001, 300_002)], "ood")
    metrics["total_time_s"] = round(time.time() - t0, 1)
    metrics["model_size_bytes"] = os.path.getsize(MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    m = main()
    for split in ("test", "ood_test"):
        for k in ("ml", "baseline"):
            print(split, k, json.dumps(m[split][k]))
    print("inference", m["inference_time_scan_gray"], "total_s", m["total_time_s"], "size", m["model_size_bytes"])

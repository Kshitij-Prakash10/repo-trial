"""Sim-to-real fine-tuning of the scanner's page segmenter.

The model is two HistGradientBoostingClassifier stages (stage 1: per-cell features; stage 2: features
plus smoothed stage-1 probabilities). Fine-tuning starts from models/scanner.joblib and ADDS boosting
iterations with warm_start; the 200 synthetic trees of each stage and the feature binning stay as
they are. The mapping from the neural-network recipe:

  Stage A "freeze backbone, train head"  -> stage 1 frozen, stage 2 (the context head) gets new trees,
                                            learning rate 0.1 (the original rate)
  Stage B "unfreeze, 10x lower LR"       -> stage 1 then stage 2 get new trees at learning rate 0.01
  early stopping on validation           -> the number of added trees is chosen on DocCorner val
  70 % real / 30 % synthetic batches     -> 70 % of the training cells come from real photos

Negatives (photos with no document) are used with an all-background mask; the document-present output
is the existing page_found (confidence >= 0.35 and a plausible quad).

CLI: python -m scanner.finetune.finetune --data-home /path/to/data [--variant full|no_stage_a|real_only]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from multiprocessing import get_context

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from scanner import dataset
from scanner import features as feats
from scanner import infer, synth
from scanner.finetune import evaluate_real as ev

SEED = 2026
REAL_CELLS = 600  # grid cells sampled per real photo
REAL_FRACTION = 0.7  # share of training cells from real photos (the rest synthetic)
LR_A, LR_B = 0.1, 0.01
STEPS_A = (25, 50, 100)  # added stage-2 trees tried in stage A
STEPS_B = (50, 100, 200)  # added trees tried in stage B (stage 1 and stage 2 each)
N_VAL = 600  # DocCorner val photos used for checkpoint selection
OUT_DIR = os.path.join(infer.ROOT, "models", "finetune")


def _rows(args: tuple[str, np.ndarray | None, int, int, str]):
    """Cached full feature map (float16 .npy path), labels, a seeded cell sample and the grid size."""
    path, corners, cells, seed, cache = args
    with Image.open(path) as f:
        img = f.convert("RGB")
    X, (gw, gh) = feats.compute(img)
    if corners is None:
        y = np.zeros(gw * gh, np.int8)
    else:
        m = Image.fromarray(((synth.poly_mask(corners, *img.size) >= 0.5) * 255).astype(np.uint8))
        y = (np.asarray(m.resize((gw, gh), Image.BOX), np.float32).ravel() >= 127.5).astype(np.int8)
    rng = np.random.default_rng([SEED, seed])
    idx = np.sort(rng.choice(len(y), size=min(cells, len(y)), replace=False))
    out = os.path.join(cache, f"{seed}.npy")
    np.save(out, X.astype(np.float16))
    return out, y, idx, (gw, gh)


def build(items: list[tuple[str, np.ndarray | None]], cells: int, seed0: int, cache: str, workers: int = 4) -> list:
    """Rows for every photo; full feature maps go to the cache folder (they are large), samples stay in memory."""
    os.makedirs(cache, exist_ok=True)
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(_rows, [(p, c, cells, seed0 + i, cache) for i, (p, c) in enumerate(items)], chunksize=16)


def _X(r) -> np.ndarray:
    return np.load(r[0], mmap_mode="r")


def real_train_items(data_home: str) -> list[tuple[str, np.ndarray | None]]:
    root = os.path.join(data_home, "prepared", "doccorner", "train")
    lab = pd.read_csv(os.path.join(root, "labels.csv"))
    return [(os.path.join(root, r["filename"]), ev._quad(r)) for r in lab.to_dict("records")]


def sim_train_items() -> list[tuple[str, np.ndarray]]:
    return [(it["image"], it["corners"]) for it in dataset.load_split("train")]


def stack(rows: list, model: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Sampled cells: X1 (features), y, and X2 (features + context from the model's stage 1) if model."""
    X1 = np.concatenate([_X(r)[r[2]] for r in rows]).astype(np.float32)
    y = np.concatenate([r[1][r[2]] for r in rows])
    if model is None:
        return X1, y, None
    ctx = []
    for r in rows:
        _, _, idx, (gw, gh) = r
        p = model["stage1"].predict_proba(np.asarray(_X(r), np.float32))[:, 1].reshape(gh, gw)
        ctx.append(feats.context(p)[idx])
    return X1, y, np.hstack([X1, np.concatenate(ctx)])


def grow(clf, X: np.ndarray, y: np.ndarray, total_iter: int, lr: float):
    """Warm-start: add trees until the classifier has total_iter of them, new ones at learning rate lr."""
    clf.set_params(warm_start=True, max_iter=total_iter, learning_rate=lr)
    return clf.fit(X, y)


def val_score(model: dict, val_items: list[dict]) -> dict:
    """Selection score on DocCorner val: mean IoU on positives minus the false-alarm rate on negatives."""
    s = ev.summarise(ev.run(model, val_items))
    s["score"] = round(s["iou_img_mean"] - (s["false_alarm_rate_on_negatives"] or 0.0), 4)
    return s


def finetune(variant: str, data_home: str, log: list) -> dict:
    """Run one variant; returns the best model by validation score (with its history in log)."""
    base = joblib.load(infer.MODEL_PATH)
    val_items = ev.doccorner_items(data_home, "val", N_VAL)
    real = real_train_items(data_home)
    t0 = time.time()
    cache = os.path.join(data_home, "feature_cache")
    rows = build(real, REAL_CELLS, 0, cache)
    n_real = sum(len(r[2]) for r in rows)
    if variant != "real_only":
        sim = sim_train_items()
        sim_cells = int(round(n_real * (1 - REAL_FRACTION) / REAL_FRACTION / len(sim)))
        rows += build(sim, sim_cells, 10_000, cache)
    t_feat = time.time() - t0
    n_sim = sum(len(r[2]) for r in rows) - n_real
    log.append({"variant": variant, "real_photos": len(real), "real_cells": n_real, "sim_cells": n_sim,
                "real_fraction": round(n_real / (n_real + n_sim), 3), "feature_seconds": round(t_feat, 1)})
    base_s = val_score(base, val_items)
    log.append({"variant": variant, "step": "start (synthetic model)", **base_s})
    best = (base_s["score"], base, "start")
    model = copy.deepcopy(base)
    n1, n2 = model["stage1"].n_iter_, model["stage2"].n_iter_
    if variant in ("full", "real_only"):  # stage A: stage 1 frozen, the context head learns
        X1, y, X2 = stack(rows, model)
        for k in STEPS_A:
            t = time.time()
            grow(model["stage2"], X2, y, n2 + k, LR_A)
            s = val_score(model, val_items)
            log.append({"variant": variant, "step": f"A +{k} stage-2 trees @ {LR_A}", "fit_s": round(time.time() - t), **s})
            if s["score"] > best[0]:
                best = (s["score"], copy.deepcopy(model), f"A+{k}")
        model = copy.deepcopy(best[1])  # the validation-chosen stage-A result (the start model if none helped)
        n2 = model["stage2"].n_iter_
    # stage B: everything learns at the lower rate
    X1, y, _ = stack(rows)
    start_b = best[2]
    for k in STEPS_B:
        t = time.time()
        m = copy.deepcopy(model)
        grow(m["stage1"], X1, y, n1 + k, LR_B)
        _, _, X2 = stack(rows, m)
        grow(m["stage2"], X2, y, n2 + k, LR_B)
        s = val_score(m, val_items)
        log.append({"variant": variant, "step": f"B +{k} trees per stage @ {LR_B}", "fit_s": round(time.time() - t), **s})
        if s["score"] > best[0]:
            best = (s["score"], m, f"{start_b} then B+{k}" if start_b != "start" else f"B+{k}")
    out = best[1]
    out = dict(out, model_version=f"scanner-0.2-ft-{variant}", data=out.get("data", "") +
               f" + real (DocCornerDataset train subset, {len(real)} photos, fine-tuned: {best[2]})")
    log.append({"variant": variant, "chosen": best[2], "val_score": best[0]})
    return out


def main(data_home: str, variants: list[str]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    threadpool_limits(4)
    for v in variants:
        log: list = []
        t = time.time()
        model = finetune(v, data_home, log)
        log.append({"variant": v, "total_seconds": round(time.time() - t)})
        joblib.dump(model, os.path.join(OUT_DIR, f"scanner_ft_{v}.joblib"), compress=3)
        with open(os.path.join(OUT_DIR, f"log_{v}.json"), "w") as f:
            json.dump(log, f, indent=1, default=str)
        print(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-home", default=os.path.join(infer.ROOT, "data", "raw"))
    ap.add_argument("--variant", action="append", choices=["full", "no_stage_a", "real_only"])
    a = ap.parse_args()
    main(a.data_home, a.variant or ["full"])

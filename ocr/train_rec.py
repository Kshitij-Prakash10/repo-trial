"""Fine-tune the text recogniser on word crops (FUNSD, CORD, SROIE train), early stopping on val CER.

    python -m ocr.train_rec --config configs/ocr/rec.json
    python -m ocr.train_rec --config configs/ocr/rec.json --overfit 48   # tiny overfit test

Starts from docTR pretrained weights. Writes checkpoints and a per-epoch log to the config's out_dir.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch

from ocr import augment
from ocr.common import DATASETS, ROOT, error_counts, rates
from ocr.data import word_crops
from ocr.models import load_rec

STORE_H = 48  # px; crops are kept at this height so the augmentations have some resolution to work on


def seed_all(seed: int) -> None:
    cv2.setNumThreads(1)  # the augmentations are tiny; leave the cores to PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _store(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    return cv2.resize(crop, (max(4, min(int(round(w * STORE_H / h)), STORE_H * 8)), STORE_H), interpolation=cv2.INTER_AREA)


def load_split(split: str, vocab: str, max_len: int, train: bool, limit_pages: int | None = None):
    """(crops, texts, dataset names). For training, words the model cannot spell are dropped (and counted)."""
    crops, texts, names, dropped = [], [], [], 0
    for ds in DATASETS:
        for crop, text, _ in word_crops(ds, split, limit_pages):
            if train and (len(text) > max_len or any(c not in vocab for c in text)):
                dropped += 1
                continue
            crops.append(_store(crop))
            texts.append(text)
            names.append(ds)
    return crops, texts, names, dropped


@torch.no_grad()
def fix_sroie_labels(model, crops, texts, names, size) -> tuple[list, list, list, dict]:
    """Clean the SROIE training labels with the pretrained recogniser (training set only).

    SROIE transcriptions are upper case whatever the print, and some derived word boxes are misaligned.
    For each SROIE crop: if the pretrained reading equals the label apart from case, the reading's case
    is used; else if the label has no letters and the reading is within half its length, the label is
    kept; otherwise the crop is dropped. Other datasets pass through unchanged."""
    idx = [i for i, n in enumerate(names) if n == "sroie"]
    model.eval()
    mean, std = model.cfg["mean"], model.cfg["std"]
    preds = []
    for i in range(0, len(idx), 256):
        out = model(to_batch([crops[j] for j in idx[i:i + 256]], size, mean, std), return_preds=True)
        preds += [p[0] for p in out["preds"]]
    new, stats = dict(), {"sroie_recased": 0, "sroie_kept_no_letters": 0, "sroie_dropped": 0}
    for j, p in zip(idx, preds):
        t = texts[j]
        if p.upper() == t.upper():
            new[j] = p
            stats["sroie_recased"] += 1
        elif not any(c.isalpha() for c in t) and error_counts(p, t)["char_err"] <= 0.5 * len(t):
            new[j] = t
            stats["sroie_kept_no_letters"] += 1
        else:
            stats["sroie_dropped"] += 1
    keep = [i for i, n in enumerate(names) if n != "sroie" or i in new]
    return [crops[i] for i in keep], [new.get(i, texts[i]) for i in keep], [names[i] for i in keep], stats


def to_batch(crops: list[np.ndarray], size: tuple[int, int], mean, std) -> torch.Tensor:
    """Resize keeping the aspect ratio, pad right and bottom with zeros, normalise (as docTR's predictor)."""
    H, W = size
    out = np.zeros((len(crops), H, W, 3), np.float32)
    for i, c in enumerate(crops):
        h, w = c.shape[:2]
        s = min(H / h, W / w)
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        out[i, :nh, :nw] = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_LINEAR) / 255.0
    out = (out - np.array(mean, np.float32)) / np.array(std, np.float32)
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def evaluate(model, crops, texts, names, size, bs: int = 256) -> dict:
    """CER / WER per dataset and their unweighted mean CER (the early-stopping metric)."""
    model.eval()
    mean, std = model.cfg["mean"], model.cfg["std"]
    preds = []
    for i in range(0, len(crops), bs):
        out = model(to_batch(crops[i:i + bs], size, mean, std), return_preds=True)
        preds += [p[0] for p in out["preds"]]
    res = {}
    for ds in sorted(set(names)):
        res[ds] = rates([error_counts(p, t, ds) for p, t, n in zip(preds, texts, names) if n == ds])
    res["mean_cer"] = round(float(np.mean([res[ds]["cer"] for ds in res])), 4)
    return res


def main(cfg: dict, overfit: int = 0) -> dict:
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 4))
    out_dir = os.path.join(ROOT, cfg["out_dir"] + ("_overfit" if overfit else ""))
    os.makedirs(out_dir, exist_ok=True)
    model = load_rec(cfg["arch"])
    size = tuple(model.cfg["input_shape"][-2:])
    mean, std = model.cfg["mean"], model.cfg["std"]
    tr_c, tr_t, tr_n, dropped = load_split("train", model.vocab, model.max_length, True, 3 if overfit else None)
    if overfit:  # a few dozen fixed samples, no augmentation, evaluated on themselves
        tr_c, tr_t, tr_n = tr_c[:overfit], tr_t[:overfit], tr_n[:overfit]
        va_c, va_t, va_n = tr_c, tr_t, tr_n
    else:
        va_c, va_t, va_n, _ = load_split("val", model.vocab, model.max_length, False)
    label_stats = {}
    if not overfit:
        tr_c, tr_t, tr_n, label_stats = fix_sroie_labels(model, tr_c, tr_t, tr_n, size)
        print("labels", json.dumps(label_stats), flush=True)
    log = {"config": cfg, "sroie_train_labels": label_stats, "overfit": overfit, "train_crops": len(tr_c), "val_crops": len(va_c),
           "train_dropped_not_in_vocab_or_too_long": dropped, "epochs": []}
    start = evaluate(model, va_c, va_t, va_n, size)
    log["start"] = start
    print("start", json.dumps(start), flush=True)
    epochs = 60 if overfit else cfg["epochs"]
    bs = min(cfg["batch_size"], len(tr_c))
    # an "epoch" is a fixed number of randomly drawn crops, so validation runs often on a CPU
    per_epoch = len(tr_c) if overfit else min(cfg.get("crops_per_epoch", len(tr_c)), len(tr_c))
    steps = epochs * (per_epoch // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=max(steps, 1), pct_start=0.1)
    rng = np.random.default_rng(cfg["seed"])
    best, bad = start["mean_cer"], 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        order = rng.permutation(len(tr_c))[:per_epoch]
        for i in range(0, len(order) - bs + 1, bs):
            idx = order[i:i + bs]
            crops = [tr_c[j] for j in idx]
            if not overfit:
                crops = [augment.phone_capture(c, rng, p=cfg["aug_p"], warp=cfg["aug_warp"], max_sigma=cfg["aug_blur"])[0]
                         for c in crops]
            loss = model(to_batch(crops, size, mean, std), [tr_t[j] for j in idx])["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if overfit and ep % 10 and ep != epochs:
            continue
        val = evaluate(model, va_c, va_t, va_n, size)
        row = {"epoch": ep, "train_loss": round(float(np.mean(losses)), 4), "val": val,
               "seconds": round(time.time() - t0), "lr": sched.get_last_lr()[0]}
        log["epochs"].append(row)
        print(json.dumps(row), flush=True)
        if not overfit:
            torch.save({"model": model.state_dict(), "arch": cfg["arch"], "epoch": ep, "val": val},
                       os.path.join(out_dir, "last.pt"))
            if val["mean_cer"] < best:
                best, bad = val["mean_cer"], 0
                torch.save({"model": model.state_dict(), "arch": cfg["arch"], "epoch": ep, "val": val},
                           os.path.join(out_dir, "best.pt"))
                log["best_epoch"] = ep
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    log["stopped_early_at"] = ep
                    break
        with open(os.path.join(out_dir, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
    log["best_val_mean_cer"] = best
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=1)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--overfit", type=int, default=0, help="train on this many fixed crops and report CER on them")
    a = ap.parse_args()
    with open(a.config) as f:
        main(json.load(f), a.overfit)

"""Fine-tune the page-orientation classifier on rotated printed and handwriting pages.

    python -m ocr.train_orient --config configs/ocr/orient.json --overfit 16   # tiny overfit test
    python -m ocr.train_orient --config configs/ocr/orient.json

Pages: FUNSD, SROIE, CORD (printed) and GNHK (handwriting), official train / val splits, plus
synthetic "pages" made by stacking IAM training lines. Every page is assumed upright as shipped and is
turned by k quarter-turns to make the label. Starts from docTR's pretrained page-orientation model.
Writes runs/orient/{best,last}.pt and log.json.
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
from PIL import Image, ImageOps

from ocr import augment, hwdata
from ocr import orientation as O
from ocr.common import DATASETS, ROOT, load_pages

STORE = 512  # px, long side of the cached pages (the model's input size)


def seed_all(seed: int) -> None:
    cv2.setNumThreads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((STORE, STORE), Image.BILINEAR)
        return np.asarray(im)


def iam_page(lines: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """A synthetic handwriting page: IAM lines stacked on white with margins (training only)."""
    n = int(rng.integers(5, 14))
    pick = [lines[int(i)] for i in rng.integers(len(lines), size=n)]
    w = max(p.shape[1] for p in pick)
    gap, m = int(rng.integers(4, 30)), int(rng.integers(20, 90))
    page = np.full((sum(p.shape[0] + gap for p in pick) + 2 * m, w + 2 * m, 3), 255, np.uint8)
    y = m
    for p in pick:
        x = m + int(rng.integers(0, max(1, w - p.shape[1] + 1)) * (rng.random() < 0.3))
        page[y:y + p.shape[0], x:x + p.shape[1]] = p[..., None]
        y += p.shape[0] + gap
    return page


def pages(split: str, limit: int | None = None) -> list[tuple[np.ndarray, str]]:
    """(page RGB, source name) for a split (train / val / test)."""
    out = []
    for ds in DATASETS:
        out += [(_load(p["path"]), ds) for p in load_pages(ds, split)[:limit]]
    out += [(_load(p["path"]), "gnhk") for p in hwdata.gnhk_pages(split)[:limit]]
    return out


def make(img: np.ndarray, k: int, rng: np.random.Generator | None, aug_p: float = 0.0) -> np.ndarray:
    if rng is not None:
        if rng.random() < 0.7:  # small skew, as left over before the deskew step
            img = O.rotate(img, float(rng.uniform(-6, 6)))
        if rng.random() < aug_p:
            img = augment.phone_capture(img, rng, p=1.0, warp=0.03, max_sigma=1.2)[0]
    return O.letterbox(np.ascontiguousarray(np.rot90(img, k)))


@torch.no_grad()
def evaluate(model, items: list[tuple[np.ndarray, str]], bs: int = 32) -> dict:
    """Accuracy over all four rotations of every page, per source and overall."""
    model.eval()
    hit: dict[str, list[int]] = {}
    xs, meta = [], []
    for img, src in items:
        for k in range(4):
            xs.append(make(img, k, None))
            meta.append((src, k))
    for i in range(0, len(xs), bs):
        pred = O.logits_k(model, O.to_tensor(xs[i:i + bs])).argmax(1).tolist()
        for (src, k), p in zip(meta[i:i + bs], pred):
            hit.setdefault(src, []).append(int(p == k))
    res = {s: round(float(np.mean(v)), 4) for s, v in hit.items()}
    res["all"] = round(float(np.mean([x for v in hit.values() for x in v])), 4)
    res["n"] = len(xs)
    return res


def main(cfg: dict, overfit: int = 0) -> dict:
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 4) if not overfit else 2)
    out_dir = os.path.join(ROOT, cfg["out_dir"] + ("_overfit" if overfit else ""))
    os.makedirs(out_dir, exist_ok=True)
    model = O.load_orient(None)
    rng = np.random.default_rng(cfg["seed"])
    train = pages("train", 4 if overfit else None)
    if overfit:
        train = train[:overfit]
        val, iam = train, []
    else:
        val = pages("val")
        iam = [np.asarray(im.convert("L").reduce(2)) for _, im, _ in hwdata.iam_lines("train", cfg["iam_lines"])]
    log = {"config": cfg, "overfit": overfit, "train_pages": len(train), "val_pages": len(val), "epochs": []}
    log["start"] = evaluate(model, val)
    print("start", json.dumps(log["start"]), flush=True)
    bs, steps_per = cfg["batch_size"], (len(train) // cfg["batch_size"] if overfit else cfg["steps_per_epoch"])
    epochs = 30 if overfit else cfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=epochs * max(steps_per, 1), pct_start=0.1)
    best, bad = log["start"]["all"], 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for _ in range(max(steps_per, 1)):
            xs, ys = [], []
            for _ in range(bs):
                if iam and rng.random() < cfg["iam_page_share"]:
                    img = iam_page(iam, rng)
                else:
                    img = train[int(rng.integers(len(train)))][0]
                k = int(rng.integers(4))
                xs.append(make(img, k, None if overfit else rng, cfg["aug_p"]))
                ys.append(k)
            loss = torch.nn.functional.cross_entropy(O.logits_k(model, O.to_tensor(xs)), torch.tensor(ys))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        if overfit and ep % 10:
            continue
        v = evaluate(model, val)
        row = {"epoch": ep, "train_loss": round(float(np.mean(losses)), 4), "val": v, "seconds": round(time.time() - t0)}
        log["epochs"].append(row)
        print(json.dumps(row), flush=True)
        if not overfit:
            state = {"model": model.state_dict(), "arch": O.ORIENT_ARCH, "epoch": ep, "val": v}
            torch.save(state, os.path.join(out_dir, "last.pt"))
            if v["all"] > best:
                best, bad, log["best_epoch"] = v["all"], 0, ep
                torch.save(state, os.path.join(out_dir, "best.pt"))
            else:
                bad += 1
        log["best_val_accuracy"] = best
        with open(os.path.join(out_dir, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
        if not overfit and bad >= cfg["patience"]:
            log["stopped_early_at"] = ep
            break
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=1)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--overfit", type=int, default=0)
    a = ap.parse_args()
    with open(a.config) as f:
        main(json.load(f), a.overfit)

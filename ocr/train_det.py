"""Fine-tune the text detector on word boxes (FUNSD, CORD, SROIE train), early stopping on val F1.

    python -m ocr.train_det --config configs/ocr/det.json
    python -m ocr.train_det --config configs/ocr/det.json --overfit 24   # tiny overfit test

Starts from docTR pretrained weights. Pages are letterboxed to the model's square input, as at inference.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from ocr import augment
from ocr.common import DATASETS, ROOT, match_boxes, prf
from ocr.data import usable, word_pages
from ocr.models import det_predictor, load_det
from ocr.train_rec import seed_all


def load_pages_boxes(split: str, limit_pages: int | None = None) -> list[dict]:
    """[{path, dataset, boxes (N, 4) px}] for every dataset."""
    out = []
    for ds in DATASETS:
        for p in word_pages(ds, split)[:limit_pages]:
            boxes = np.array([it["box"] for it in p["items"] if usable(it)], np.float32).reshape(-1, 4)
            out.append({"path": p["path"], "dataset": ds, "boxes": boxes})
    return out


def letterbox(img: np.ndarray, boxes: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize keeping the aspect ratio, pad symmetrically with zeros. Boxes come back relative (0..1)."""
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    canvas = np.zeros((size, size, 3), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    b = boxes * s + np.array([ox, oy, ox, oy], np.float32)
    b = np.clip(b, 0, size)
    keep = (b[:, 2] - b[:, 0] >= 2) & (b[:, 3] - b[:, 1] >= 2)
    return canvas, (b[keep] / size).astype(np.float32)


def make_batch(items: list[dict], size: int, mean, std, rng, aug: dict | None):
    xs, targets = [], []
    for it in items:
        with Image.open(it["path"]) as im:
            img = np.asarray(im.convert("RGB"))
        boxes = it["boxes"]
        if aug:
            img, boxes = augment.phone_capture(img, rng, boxes, p=aug["p"], warp=aug["warp"], max_sigma=aug["blur"])
        img, rel = letterbox(img, np.asarray(boxes, np.float32).reshape(-1, 4), size)
        xs.append((img / 255.0 - np.array(mean)) / np.array(std))
        targets.append({"words": rel})
    x = torch.from_numpy(np.stack(xs).astype(np.float32)).permute(0, 3, 1, 2).contiguous()
    return x, targets


def freeze_batchnorm(model) -> None:
    """Keep the pretrained BatchNorm statistics: batches of 2 pages are too small to re-estimate them."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def evaluate(model, pages: list[dict]) -> dict:
    """Word-box precision / recall / F1 at IoU 0.5 per dataset, and the unweighted mean F1."""
    pred = det_predictor(model)
    counts = {}
    for i in range(0, len(pages), 8):
        chunk = pages[i:i + 8]
        imgs = []
        for it in chunk:
            with Image.open(it["path"]) as im:
                imgs.append(np.asarray(im.convert("RGB")))
        for it, img, out in zip(chunk, imgs, pred(imgs)):
            H, W = img.shape[:2]
            b = out["words"][:, :4] * np.array([W, H, W, H])
            c = counts.setdefault(it["dataset"], [0, 0, 0])
            c[0] += len(match_boxes(b, it["boxes"]))
            c[1] += len(b)
            c[2] += len(it["boxes"])
    res = {ds: prf(*c) for ds, c in counts.items()}
    res["mean_f1"] = round(float(np.mean([res[ds]["f1"] for ds in counts])), 4)
    return res


def main(cfg: dict, overfit: int = 0) -> dict:
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 4))
    out_dir = os.path.join(ROOT, cfg["out_dir"] + ("_overfit" if overfit else ""))
    os.makedirs(out_dir, exist_ok=True)
    model = load_det(cfg["arch"])
    size = model.cfg["input_shape"][-1]
    mean, std = model.cfg["mean"], model.cfg["std"]
    if overfit:  # a few dozen fixed pages, no augmentation, evaluated on themselves
        per = max(1, overfit // len(DATASETS))
        train = load_pages_boxes("train", per)
        val = train
    else:
        train, val = load_pages_boxes("train"), load_pages_boxes("val")
    log = {"config": cfg, "overfit": overfit, "train_pages": len(train), "val_pages": len(val), "epochs": []}
    start = evaluate(model, val)
    log["start"] = start
    print("start", json.dumps(start), flush=True)
    bs = cfg["batch_size"]
    per_epoch = len(train) if overfit else min(cfg["pages_per_epoch"], len(train))
    epochs = 12 if overfit else cfg["epochs"]
    steps = epochs * (per_epoch // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=max(steps, 1), pct_start=0.1)
    rng = np.random.default_rng(cfg["seed"])
    aug = None if overfit else {"p": cfg["aug_p"], "warp": cfg["aug_warp"], "blur": cfg["aug_blur"]}
    best, bad = start["mean_f1"], 0
    for ep in range(1, epochs + 1):
        model.train()
        freeze_batchnorm(model)
        t0, losses = time.time(), []
        order = rng.permutation(len(train))[:per_epoch]
        for i in range(0, len(order) - bs + 1, bs):
            x, targets = make_batch([train[j] for j in order[i:i + bs]], size, mean, std, rng, aug)
            loss = model(x, targets)["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if overfit and ep % 4 and ep != epochs:
            print(f"epoch {ep} train_loss {np.mean(losses):.4f}", flush=True)
            continue
        v = evaluate(model, val)
        row = {"epoch": ep, "train_loss": round(float(np.mean(losses)), 4), "val": v,
               "seconds": round(time.time() - t0), "lr": sched.get_last_lr()[0]}
        log["epochs"].append(row)
        print(json.dumps(row), flush=True)
        if not overfit:
            torch.save({"model": model.state_dict(), "arch": cfg["arch"], "epoch": ep, "val": v},
                       os.path.join(out_dir, "last.pt"))
            if v["mean_f1"] > best:
                best, bad = v["mean_f1"], 0
                torch.save({"model": model.state_dict(), "arch": cfg["arch"], "epoch": ep, "val": v},
                           os.path.join(out_dir, "best.pt"))
                log["best_epoch"] = ep
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    log["stopped_early_at"] = ep
                    break
        with open(os.path.join(out_dir, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
    log["best_val_mean_f1"] = best
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=1)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--overfit", type=int, default=0, help="train on about this many fixed pages and report F1 on them")
    a = ap.parse_args()
    with open(a.config) as f:
        main(json.load(f), a.overfit)

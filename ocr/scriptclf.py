"""Printed-vs-handwritten classifier for word crops (routes text to the right recogniser).

    python -m ocr.scriptclf --config configs/ocr/scriptclf.json            # train
    python -m ocr.scriptclf --config configs/ocr/scriptclf.json --overfit 64

A small CNN on grey 32 x 96 word crops. Handwritten examples: GNHK handwritten words and windows cut
from IAM lines. Printed examples: SROIE and CORD words and GNHK printed words. FUNSD is not used for
training because its forms mix both kinds without labels. Writes runs/scriptclf/{best,last}.pt.
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
from torch import nn

from ocr import hwdata
from ocr.common import ROOT
from ocr.data import word_crops

CKPT = os.path.join(ROOT, "checkpoints", "scriptclf_v1", "scriptclf.pt")
H, W = 32, 96


def prep(crop: np.ndarray) -> np.ndarray:
    """Grey, height 32, aspect kept, right-padded (or squeezed) to width 96, contrast stretched. uint8."""
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    h, w = g.shape
    nw = int(np.clip(round(w * H / max(h, 1)), 8, W))
    g = cv2.resize(g, (nw, H), interpolation=cv2.INTER_AREA if h > H else cv2.INTER_LINEAR)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    g = np.clip((g.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    if np.median(g) < 110:  # light text on dark: invert so ink is dark
        g = 255 - g
    out = np.full((H, W), int(np.median(np.concatenate([g[0], g[-1]]))), np.uint8)
    out[:, :nw] = g
    return out


class ScriptNet(nn.Module):
    """About 0.1 M parameters; output is one logit (positive = handwritten)."""

    def __init__(self) -> None:
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 96))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


def to_tensor(crops: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(crops).astype(np.float32) / 255.0 - 0.5).unsqueeze(1)


def load_clf(ckpt: str = CKPT) -> ScriptNet:
    model = ScriptNet()
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return model.eval()


@torch.no_grad()
def handwritten_prob(model: ScriptNet, crops: list[np.ndarray], bs: int = 256) -> np.ndarray:
    """P(handwritten) for RGB word crops."""
    if not crops:
        return np.zeros(0)
    x = [prep(c) for c in crops]
    return np.concatenate([torch.sigmoid(model(to_tensor(x[i:i + bs]))).numpy() for i in range(0, len(x), bs)])


def _poly_crop(img: np.ndarray, poly: list) -> np.ndarray | None:
    p = np.array(poly)
    x0, y0, x1, y1 = max(0, int(p[:, 0].min())), max(0, int(p[:, 1].min())), int(p[:, 0].max()), int(p[:, 1].max())
    c = img[y0:y1, x0:x1]
    return c if c.shape[0] >= 6 and c.shape[1] >= 6 else None


def load_words(split: str, limit: int | None = None, per_printed: int = 12000, seed: int = 2026) -> dict[str, list]:
    """Prepared crops per source: {"gnhk_hw", "iam_hw", "gnhk_pr", "sroie_pr", "cord_pr"} for train / val / test."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    out: dict[str, list] = {k: [] for k in ("gnhk_hw", "iam_hw", "gnhk_pr", "sroie_pr", "cord_pr")}
    for page in hwdata.gnhk_pages(split)[:limit]:
        img = hwdata.load_image(page["path"])
        for w in page["words"]:
            if w["text"] in hwdata.GNHK_SPECIAL:
                continue
            c = _poly_crop(img, w["poly"])
            if c is not None:
                out["gnhk_hw" if w["type"] == "H" else "gnhk_pr"].append(prep(c))
    for _, im, text in hwdata.iam_lines("validation" if split == "val" else split, None if limit is None else limit * 10):
        a = np.asarray(im.convert("L"))
        for _ in range(3):  # windows about as wide as a word
            ww = int(a.shape[0] * rng.uniform(0.8, 4.0))
            x = int(rng.integers(0, max(1, a.shape[1] - ww)))
            out["iam_hw"].append(prep(a[:, x:x + ww]))
    for ds in ("sroie", "cord"):
        crops = [c for c, _, _ in word_crops(ds, split, limit)]
        pick = rng.permutation(len(crops))[:per_printed]
        out[f"{ds}_pr"] = [prep(crops[i]) for i in pick]
    return out


@torch.no_grad()
def evaluate(model: ScriptNet, sets: dict[str, list]) -> dict:
    model.eval()
    res = {}
    for name, crops in sets.items():
        if not crops:
            continue
        p = np.concatenate([torch.sigmoid(model(to_tensor(crops[i:i + 512]))).numpy() for i in range(0, len(crops), 512)])
        res[name] = round(float(((p >= 0.5) == name.endswith("_hw")).mean()), 4)
    res["mean"] = round(float(np.mean(list(res.values()))), 4)  # unweighted over sources
    return res


def _augment(c: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.3:
        c = cv2.GaussianBlur(c, (0, 0), float(rng.uniform(0.4, 1.2)))
    if rng.random() < 0.3:
        c = np.clip(c.astype(np.float32) * rng.uniform(0.6, 1.0) + rng.uniform(0, 60), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        c = np.clip(c + rng.normal(0, rng.uniform(2, 12), c.shape), 0, 255).astype(np.uint8)
    return c


def main(cfg: dict, overfit: int = 0) -> dict:
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 2))
    out_dir = os.path.join(ROOT, cfg["out_dir"] + ("_overfit" if overfit else ""))
    os.makedirs(out_dir, exist_ok=True)
    train = load_words("train", 3 if overfit else None)
    if overfit:
        train = {k: v[:overfit // 4] for k, v in train.items()}
        val = train
    else:
        val = load_words("val")
    model = ScriptNet()
    rng = np.random.default_rng(cfg["seed"])
    log = {"config": cfg, "overfit": overfit, "train": {k: len(v) for k, v in train.items()},
           "val": {k: len(v) for k, v in val.items()}, "epochs": []}
    hw = [k for k in train if k.endswith("_hw") and train[k]]
    pr = [k for k in train if k.endswith("_pr") and train[k]]
    epochs, bs = (40 if overfit else cfg["epochs"]), cfg["batch_size"]
    steps = 20 if overfit else cfg["steps_per_epoch"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=epochs * steps, pct_start=0.1)
    best, bad = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for _ in range(steps):
            xs, ys = [], []
            for i in range(bs):  # half handwritten, half printed; sources drawn evenly within each half
                srcs = hw if i % 2 == 0 else pr
                s = srcs[int(rng.integers(len(srcs)))]
                c = train[s][int(rng.integers(len(train[s])))]
                xs.append(c if overfit else _augment(c, rng))
                ys.append(float(i % 2 == 0))
            loss = nn.functional.binary_cross_entropy_with_logits(model(to_tensor(xs)), torch.tensor(ys))
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
            state = {"model": model.state_dict(), "epoch": ep, "val": v}
            torch.save(state, os.path.join(out_dir, "last.pt"))
            if v["mean"] > best:
                best, bad, log["best_epoch"] = v["mean"], 0, ep
                torch.save(state, os.path.join(out_dir, "best.pt"))
            else:
                bad += 1
        log["best_val_mean_accuracy"] = best
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

"""Fine-tune the handwriting recogniser (TrOCR) with a curriculum and printed-text replay.

    python -m ocr.train_hw --config configs/ocr/hw.json --overfit 32     # tiny overfit test
    python -m ocr.train_hw --config configs/ocr/hw.json                  # full run (hours on a CPU)
    python -m ocr.train_hw --config configs/ocr/hw.json --resume         # continue after an interruption

Sources mixed in every batch, with shares that change over three phases (the curriculum):
  synth    synthetic handwriting (ocr/synth_hw.py), text taken from training transcriptions only
  iam      IAM training lines
  gnhk     GNHK training lines (photographed handwriting)
  printed  printed replay: GNHK printed lines and synthetic lines in system fonts
Validation: IAM validation lines and GNHK validation lines; early stopping on their mean CER.
Writes runs/hw/{best,last}/ (Hugging Face format), log.json and the config.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import time

import cv2
import numpy as np
import torch

from ocr import augment, hwdata, synth_hw
from ocr.common import ROOT, error_counts, rates
from ocr.handwriting import HW_NAME, MAX_TOKENS, HandwritingReader, prep_line

SYSTEM_FONTS = ("/usr/share/fonts/truetype/dejavu/*.ttf", "/usr/share/fonts/truetype/liberation/*.ttf")


def seed_all(seed: int) -> None:
    cv2.setNumThreads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _gray(crop: np.ndarray) -> np.ndarray:
    return prep_line(crop)[..., 0].copy()  # a copy: a view would keep all three channels alive


def load_real(split: str, limit: int | None = None) -> dict[str, list[tuple[np.ndarray, str]]]:
    """Prepared grey lines per source for a split (train or val)."""
    out = {"iam": [], "gnhk": [], "printed": []}
    for _, im, text in hwdata.iam_lines("train" if split == "train" else "validation", limit):
        out["iam"].append((_gray(np.asarray(im)), text))
    for page in hwdata.gnhk_pages(split)[:limit]:
        for ln in hwdata.gnhk_lines(page):
            if ln["special"] or not ln["text"].strip() or len(ln["text"]) > 90:
                continue
            out["printed" if ln["type"] == "P" else "gnhk"].append((_gray(ln["crop"]), ln["text"]))
    return out


def augment_line(g: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    """Light phone-capture augmentation on a prepared grey line; returns RGB."""
    rgb = np.repeat(g[..., None], 3, 2)
    if rng.random() < p:
        rgb = augment.phone_capture(rgb, rng, p=1.0, warp=0.03, max_sigma=1.0)[0]
    return rgb


def teacher_forced_loss(model, tok, px: torch.Tensor, texts: list[str]) -> torch.Tensor:
    """Cross-entropy with the pretrained model's own convention: the decoder starts from </s> and
    predicts the text tokens then </s> (no <s>). Done by hand: the library's built-in label path
    shifts the labels a second time for this decoder and starts from a loss above 10."""
    ids = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS).input_ids
    target = ids[:, 1:]  # drop <s>; keeps the tokens, </s> and padding
    start = torch.full((len(ids), 1), model.generation_config.decoder_start_token_id, dtype=ids.dtype)
    logits = model(pixel_values=px, decoder_input_ids=torch.cat([start, target[:, :-1]], 1)).logits
    mask = target != tok.pad_token_id
    return torch.nn.functional.cross_entropy(logits[mask], target[mask])


@torch.no_grad()
def evaluate(reader: HandwritingReader, sets: dict[str, list[tuple[np.ndarray, str]]]) -> dict:
    reader.model.eval()
    res = {}
    for name, items in sets.items():
        preds = reader.read([np.repeat(g[..., None], 3, 2) for g, _ in items], prep=False)
        res[name] = rates([error_counts(p[0], t) for p, (_, t) in zip(preds, items)])
    res["mean_cer"] = round(float(np.mean([res[k]["cer"] for k in sets])), 4)
    return res


def main(cfg: dict, overfit: int = 0, resume: bool = False) -> dict:
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 4))
    out_dir = os.path.join(ROOT, cfg["out_dir"] + ("_overfit" if overfit else ""))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.json")
    last = os.path.join(out_dir, "last")
    start_step, log = 0, None
    if resume and os.path.exists(os.path.join(last, "config.json")) and os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
        start_step = log["evals"][-1]["step"] if log["evals"] else 0
    reader = HandwritingReader(last if start_step else cfg.get("init", HW_NAME), revision=cfg.get("revision"))
    model, tok = reader.model, reader.processor.tokenizer
    rng = np.random.default_rng(cfg["seed"] + start_step)
    train = load_real("train", 4 if overfit else None)
    texts = [t for src in ("iam", "gnhk") for _, t in train[src]]  # corpus for the synthetic lines
    hw_fonts, pr_fonts = synth_hw.font_files(), sorted(f for pat in SYSTEM_FONTS for f in glob.glob(pat))
    if overfit:
        fixed = (train["iam"] + train["gnhk"])[:overfit]
        val = {"overfit": fixed}
    else:
        v = load_real("val")
        pick = np.random.default_rng(cfg["seed"]).permutation(len(v["iam"]))[:cfg["val_iam_lines"]]
        val = {"iam": [v["iam"][i] for i in pick], "gnhk": v["gnhk"][:cfg["val_gnhk_lines"]]}
    if log is None:
        log = {"config": cfg, "overfit": overfit, "train_lines": {k: len(v) for k, v in train.items()},
               "val_lines": {k: len(v) for k, v in val.items()}, "fonts": {"handwriting": len(hw_fonts), "printed": len(pr_fonts)},
               "evals": []}
        log["start"] = evaluate(reader, val)
        print("start", json.dumps(log["start"]), flush=True)
    steps = 150 if overfit else cfg["steps"]
    bs = cfg["batch_size"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    for _ in range(start_step):
        sched.step()
    best = min([log["start"]["mean_cer"]] + [e["val"]["mean_cer"] for e in log["evals"]])
    bad, losses, t0 = 0, [], time.time()
    every = 50 if overfit else cfg["eval_every"]

    def sample(step: int) -> tuple[np.ndarray, str]:
        if overfit:
            raise RuntimeError("unused in overfit mode")
        phase = next(ph for ph in cfg["curriculum"] if step < ph["until"] * steps)
        src = rng.choice(list(phase["mix"]), p=np.array(list(phase["mix"].values())) / sum(phase["mix"].values()))
        if src == "synth" or (src == "printed" and rng.random() < 0.5):
            text = texts[int(rng.integers(len(texts)))]
            img = synth_hw.render_line(text, rng, hw_fonts if src == "synth" else pr_fonts)
            return prep_line(img), text
        g, text = train[src][int(rng.integers(len(train[src])))]
        return augment_line(g, rng, cfg["aug_p"]), text

    for step in range(start_step, steps):
        model.train()
        if overfit:  # fixed pairs, no augmentation
            idx = rng.integers(len(fixed), size=min(bs, len(fixed)))
            imgs, txts = [np.repeat(fixed[i][0][..., None], 3, 2) for i in idx], [fixed[i][1] for i in idx]
        else:
            imgs, txts = zip(*[sample(step) for _ in range(bs)])
        px = reader.processor(images=list(imgs), return_tensors="pt").pixel_values
        loss = teacher_forced_loss(model, tok, px, list(txts))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % every and step + 1 != steps:
            continue
        v = evaluate(reader, val)
        row = {"step": step + 1, "train_loss": round(float(np.mean(losses)), 4), "val": v,
               "seconds": round(time.time() - t0), "lr": sched.get_last_lr()[0]}
        losses, t0 = [], time.time()
        log["evals"].append(row)
        print(json.dumps(row), flush=True)
        if not overfit:
            for d in (last,) + ((os.path.join(out_dir, "best"),) if v["mean_cer"] < best else ()):
                shutil.rmtree(d + ".tmp", ignore_errors=True)
                model.save_pretrained(d + ".tmp")
                reader.processor.save_pretrained(d + ".tmp")
                shutil.rmtree(d, ignore_errors=True)
                os.replace(d + ".tmp", d)
            if v["mean_cer"] < best:
                best, bad, log["best_step"] = v["mean_cer"], 0, step + 1
            else:
                bad += 1
        log["best_val_mean_cer"] = best
        with open(log_path, "w") as f:
            json.dump(log, f, indent=1)
        if not overfit and bad >= cfg["patience"]:
            log["stopped_early_at"] = step + 1
            break
    with open(log_path, "w") as f:
        json.dump(log, f, indent=1)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--overfit", type=int, default=0, help="train on this many fixed lines and report CER on them")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    with open(a.config) as f:
        main(json.load(f), a.overfit, a.resume)

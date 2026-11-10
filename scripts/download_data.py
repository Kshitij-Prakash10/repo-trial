#!/usr/bin/env python3
"""Download the public OCR datasets and convert them to one common annotation format.

    python scripts/download_data.py                 # all datasets
    python scripts/download_data.py --only funsd    # one dataset
    python scripts/download_data.py --check         # verify what is on disk, download nothing

Sources (Hugging Face, pinned revisions; none needs registration):
  funsd   nielsr/funsd             FUNSD forms, word boxes.     Licence: non-commercial research / education.
  sroie   jsdnrs/ICDAR2019-SROIE   SROIE receipts, LINE boxes.  Licence: CC BY 4.0.
  cord    naver-clova-ix/cord-v2   CORD receipts, word quads.   Licence: CC BY 4.0.

Output, per dataset, under data/<name>/ (never committed):
  images/<split>/<id>.<ext>        the page image, bytes exactly as published
  <split>.jsonl                    one page per line:
      {"id", "dataset", "split", "image" (path relative to data/<name>/), "width", "height",
       "level": "word" | "line",   what one item is (SROIE is annotated per text line)
       "items": [{"text", "box": [x0, y0, x1, y1] in pixels, "quad": 4 x [x, y] or null,
                  "line": int or null (the dataset's own line / row id, when it has one)}]}
  splits.json                      the page ids of every split, and how the split was made

Splits: the official test split is kept as it is and never used for training. CORD has an official
validation split. FUNSD and SROIE have none, so a fixed, seeded part of their official train split is
set aside as validation (SEED below).

Disk: the free space is checked before every download, and the script stops when the data would not fit
or when the total for all datasets would pass MAX_TOTAL_GB.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "_hf_cache")
SEED = 2026
MAX_TOTAL_GB = 30.0  # stop and ask before going over this
MARGIN_GB = 2.0  # free space that must remain after a download and its conversion

DATASETS = {
    "funsd": {
        "repo": "nielsr/funsd", "revision": "7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5", "download_gb": 0.02,
        "files": {"train": ["data/train-00000-of-00001.parquet"], "test": ["data/test-00000-of-00001.parquet"]},
        "level": "word", "val_from_train": 20,
        "licence": "FUNSD licence: non-commercial, research and educational use",
    },
    "sroie": {
        "repo": "jsdnrs/ICDAR2019-SROIE", "revision": "bffe40c26759f3376ec2b3ae9031dbba54cd587c", "download_gb": 0.51,
        "files": {"train": ["data/train-00000-of-00001.parquet"], "test": ["data/test-00000-of-00001.parquet"]},
        "level": "line", "val_from_train": 60,
        "licence": "CC BY 4.0",
    },
    "cord": {
        "repo": "naver-clova-ix/cord-v2", "revision": "7f0115a4b758a71d6473b8d085751692da2fef98", "download_gb": 2.31,
        "files": {"train": [f"data/train-0000{i}-of-00004-{h}.parquet" for i, h in enumerate(
                      ["b4aaeceff1d90ecb", "7dbbe248962764c5", "688fe1305a55e5cc", "2d0cd200555ed7fd"])],
                  "val": ["data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet"],
                  "test": ["data/test-00000-of-00001-9c204eb3f4e11791.parquet"]},
        "level": "word", "val_from_train": 0,
        "licence": "CC BY 4.0",
    },
}


def free_gb(path: str = DATA) -> float:
    os.makedirs(path, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def check_space(name: str, need_gb: float, planned_total_gb: float) -> None:
    """Stop (exit code 2) when a download would not fit or the total would pass MAX_TOTAL_GB."""
    if planned_total_gb > MAX_TOTAL_GB:
        sys.exit(f"STOP: downloading {name} would bring the data to {planned_total_gb:.1f} GB, over the "
                 f"{MAX_TOTAL_GB:.0f} GB limit. Ask before continuing.")
    free = free_gb()
    # the parquet cache plus the extracted images: about twice the download
    if free < 2 * need_gb + MARGIN_GB:
        sys.exit(f"STOP: {name} needs about {2 * need_gb:.1f} GB plus {MARGIN_GB:.0f} GB margin, "
                 f"but only {free:.1f} GB is free.")
    print(f"[{name}] free disk {free:.1f} GB, download {need_gb:.2f} GB: ok", flush=True)


def _ext(data: bytes) -> str:
    return "png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"


def _box(points) -> list[int]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _items_funsd(r: dict, w: int, h: int) -> list[dict]:
    # this copy stores boxes on a 0..1000 scale in both directions (checked by drawing them); back to pixels
    return [{"text": t, "box": [round(b[0] * w / 1000), round(b[1] * h / 1000),
                                round(b[2] * w / 1000), round(b[3] * h / 1000)], "quad": None, "line": None}
            for t, b in zip(r["words"], r["bboxes"]) if t.strip()]


def _items_sroie(r: dict, w: int, h: int) -> list[dict]:
    return [{"text": t.strip(), "box": [int(v) for v in b], "quad": None, "line": i}
            for i, (t, b) in enumerate(zip(r["words"], r["bboxes"])) if t.strip()]


def _items_cord(r: dict, w: int, h: int) -> list[dict]:
    gt = json.loads(r["ground_truth"])
    out = []
    for ln in gt["valid_line"]:
        for w in ln["words"]:
            if not w["text"].strip():
                continue
            q = w["quad"]
            quad = [[q[f"x{i}"], q[f"y{i}"]] for i in (1, 2, 3, 4)]
            out.append({"text": w["text"].strip(), "box": _box(quad), "quad": quad, "line": int(w["row_id"])})
    return out


ITEMS = {"funsd": _items_funsd, "sroie": _items_sroie, "cord": _items_cord}


def _download(cfg: dict, f: str, tries: int = 4) -> str:
    """One file from the pinned revision; a dropped connection is retried (the download resumes)."""
    import time
    from huggingface_hub import hf_hub_download
    for k in range(tries):
        try:
            return hf_hub_download(cfg["repo"], f, repo_type="dataset", revision=cfg["revision"], cache_dir=CACHE)
        except Exception as e:  # network errors only matter here; the last one is re-raised
            if k == tries - 1:
                raise
            print(f"  retry {k + 1} after: {type(e).__name__}", flush=True)
            time.sleep(2 ** (k + 1))


def convert(name: str) -> dict:
    """Download one dataset (if needed) and write images, <split>.jsonl and splits.json."""
    import pyarrow.parquet as pq
    from PIL import Image

    cfg = DATASETS[name]
    out = os.path.join(DATA, name)
    pages: dict[str, list[dict]] = {}
    for split, files in cfg["files"].items():
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        pages[split] = []
        n = 0
        for f in files:
            path = _download(cfg, f)
            pf = pq.ParquetFile(path)
            for g in range(pf.num_row_groups):
                for r in pf.read_row_group(g).to_pylist():
                    raw = r["image"]["bytes"]
                    pid = str(r.get("id") or r.get("key") or f"{split}_{n:04d}")
                    if name in ("cord", "funsd"):  # no usable ids (FUNSD's restart at 0 in every split)
                        pid = f"{split}_{n:04d}"
                    n += 1
                    rel = f"images/{split}/{pid}.{_ext(raw)}"
                    with open(os.path.join(out, rel), "wb") as fh:
                        fh.write(raw)
                    with Image.open(io.BytesIO(raw)) as im:
                        w, h = im.size
                    pages[split].append({"id": pid, "dataset": name, "split": split, "image": rel, "width": w,
                                         "height": h, "level": cfg["level"], "items": ITEMS[name](r, w, h)})
    how = {"test": "official test split", "train": "official train split"}
    if cfg["val_from_train"]:
        # a fixed part of the official train split becomes validation (seeded, independent of file order)
        ids = sorted(p["id"] for p in pages["train"])
        random.Random(SEED).shuffle(ids)
        val_ids = set(ids[:cfg["val_from_train"]])
        pages["val"] = [dict(p, split="val") for p in pages["train"] if p["id"] in val_ids]
        pages["train"] = [p for p in pages["train"] if p["id"] not in val_ids]
        how["val"] = f"{cfg['val_from_train']} pages taken from the official train split, seed {SEED}"
        how["train"] = "official train split minus the validation pages"
    else:
        how["val"] = "official validation split"
    summary = {"dataset": name, "source": cfg["repo"], "revision": cfg["revision"], "licence": cfg["licence"],
               "level": cfg["level"], "how": how, "pages": {}, "items": {}, "ids": {}}
    for split, ps in pages.items():
        ps.sort(key=lambda p: p["id"])
        with open(os.path.join(out, f"{split}.jsonl"), "w") as fh:
            for p in ps:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        summary["pages"][split] = len(ps)
        summary["items"][split] = sum(len(p["items"]) for p in ps)
        summary["ids"][split] = [p["id"] for p in ps]
    test, train, val = (set(summary["ids"][s]) for s in ("test", "train", "val"))
    if test:
        assert not (test & train) and not (test & val) and not (train & val), "split overlap"
    summary["annotation_sha256"] = {s: hashlib.sha256(open(os.path.join(out, f"{s}.jsonl"), "rb").read()).hexdigest()
                                    for s in pages}
    with open(os.path.join(out, "splits.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def check(name: str) -> dict | None:
    p = os.path.join(DATA, name, "splits.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    for split, n in s["pages"].items():
        have = sum(1 for _ in open(os.path.join(DATA, name, f"{split}.jsonl")))
        assert have == n, f"{name}/{split}: {have} pages on disk, {n} expected"
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", choices=sorted(DATASETS))
    ap.add_argument("--check", action="store_true", help="verify what is on disk; download nothing")
    a = ap.parse_args()
    names = a.only or list(DATASETS)
    total = 0.0
    for name in names:
        if not a.check and check(name) is None:
            # count the cache and the extracted images towards the limit
            total += 2 * DATASETS[name]["download_gb"]
            check_space(name, DATASETS[name]["download_gb"], total)
            convert(name)
        s = check(name)
        if s is None:
            print(f"[{name}] not downloaded")
        else:
            print(f"[{name}] pages {s['pages']} items {s['items']} ({s['level']} level, {s['licence']})", flush=True)

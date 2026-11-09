"""Evaluate scanner models on every suite and write per-image results and summaries.

Suites: smartdoc (SmartDoc 2015 Ch1, held-out test, every 10th frame), doccorner_val, doccorner_test,
synthetic_test (the original synthetic test set, to check that fine-tuning did not forget it).
CLI: python -m scanner.finetune.report --data-home DIR --name base --model models/scanner.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from scanner import dataset, infer
from scanner.finetune import evaluate_real as ev
from scanner.finetune.prep_real import MANIFEST_DIR

EVAL_DIR = os.path.join(infer.ROOT, "models", "finetune", "eval")
SUITES = ("smartdoc", "doccorner_val", "doccorner_test", "synthetic_test")


def synthetic_items() -> list[dict]:
    out = []
    for it in dataset.load_split("test"):
        c = it["corners"]
        row = {f"{k}_{a}": float(c[i][j]) for i, k in enumerate(ev.CORNERS) for j, a in enumerate("xy")}
        out.append({"id": os.path.basename(it["image"]), "path": it["image"], "row": {**row, **it["meta"]},
                    "kind": "synthetic", "scale": 1.0})
    return out


def items_for(suite: str, data_home: str) -> list[dict]:
    if suite == "smartdoc":
        return ev.smartdoc_items(data_home, os.path.join(MANIFEST_DIR, "smartdoc_ch1_test_frames.csv"))
    if suite == "doccorner_val":
        return ev.doccorner_items(data_home, "val")
    if suite == "doccorner_test":
        return ev.doccorner_items(data_home, "test")
    return synthetic_items()


def breakdown(df: pd.DataFrame, items: list[dict], suite: str) -> dict:
    """Success rates by condition where the metadata allows it."""
    meta = pd.DataFrame([{"id": it["id"], **{k: it["row"].get(k) for k in
                                             ("bg_name", "modeltype_name", "source", "background")}}
                         for it in items])
    d = df.merge(meta, on="id")
    d = d[~d.negative]
    key = "iou_doc" if suite == "smartdoc" else "iou_img"
    out = {}
    for col in ("bg_name", "modeltype_name", "source", "background"):
        if d[col].notna().any():
            g = d.groupby(col)[key]
            out[col] = {k: {"n": int(v.size), "iou_mean": round(float(v.mean()), 4),
                            "iou_ge_0.9": round(float((v >= 0.9).mean()), 4)} for k, v in g}
    return out


def latency(model: dict, items: list[dict], n: int = 40) -> dict:
    """Wall time of infer.detect per photo in one process (model loaded), SmartDoc full-HD frames."""
    ts = []
    for it in items[:n]:
        with Image.open(it["path"]) as f:
            img = f.convert("RGB")
        t = time.perf_counter()
        infer.detect(img, model)
        ts.append((time.perf_counter() - t) * 1000)
    return {"n": len(ts), "ms_mean": round(float(np.mean(ts)), 1), "ms_p90": round(float(np.percentile(ts, 90)), 1),
            "image": "1920x1080 SmartDoc frame", "threads": "default"}


def evaluate_model(name: str, model_path: str, data_home: str, suites=SUITES) -> dict:
    model = joblib.load(model_path)
    os.makedirs(EVAL_DIR, exist_ok=True)
    out = {"name": name, "model": os.path.relpath(model_path, infer.ROOT),
           "model_version": model.get("model_version"), "size_bytes": os.path.getsize(model_path),
           "trees": [model["stage1"].n_iter_, model["stage2"].n_iter_ if model.get("stage2") else 0]}
    for suite in suites:
        items = items_for(suite, data_home)
        t = time.time()
        df = ev.run(model, items)
        df.to_csv(os.path.join(EVAL_DIR, f"{name}__{suite}.csv.gz"), index=False)
        out[suite] = {**ev.summarise(df, smartdoc=suite == "smartdoc"), "seconds": round(time.time() - t),
                      "by_condition": breakdown(df, items, suite)}
        print(name, suite, json.dumps({k: v for k, v in out[suite].items() if k != "by_condition"}), flush=True)
    out["latency"] = latency(model, items_for("smartdoc", data_home))
    with open(os.path.join(EVAL_DIR, f"{name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-home", default=os.path.join(infer.ROOT, "data", "raw"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", action="append", choices=SUITES)
    a = ap.parse_args()
    evaluate_model(a.name, a.model, a.data_home, a.suite or SUITES)

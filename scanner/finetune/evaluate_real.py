"""Evaluate a scanner model on real photos: DocCornerDataset (prepared subsets) and SmartDoc 2015 Ch1.

Metrics per image (corners in image px, TL TR BR BL; negatives have no corners):
  iou_img       IoU of the predicted and true quads in the image plane (pixel masks at full resolution)
  iou_doc       SmartDoc only: the official Challenge-1 Jaccard index, computed in the document's own frame
                (both quads mapped by the homography that sends the true quad to the model_width x
                model_height rectangle), as in the competition evaluation
  err_px        mean corner distance in original-image pixels, best cyclic / reversed correspondence
  err_pct_img   err_px as a percentage of the image diagonal
  found         page_found (the model's document-present output); on negatives found is a false alarm
Prediction: scanner.infer.detect on the stored image; a fallback counts with the image corners.
"""
from __future__ import annotations

import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from scanner import infer, synth

CORNERS = ("tl", "tr", "br", "bl")
_MODEL: dict | None = None


def _quad(r: dict) -> np.ndarray | None:
    if r.get("tl_x") is None or pd.isna(r.get("tl_x")):
        return None
    return np.array([[r[f"{c}_x"], r[f"{c}_y"]] for c in CORNERS], dtype=float)


def match_err(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean corner distance (px) under the best cyclic and reversed correspondence."""
    best = np.inf
    for t in (true, true[::-1]):
        for s in range(4):
            best = min(best, float(np.linalg.norm(pred - np.roll(t, s, 0), axis=1).mean()))
    return best


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip(subject: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon by an axis-aligned rectangle."""
    def edge(poly, inside, cut):
        out = []
        for i in range(len(poly)):
            a, b = poly[i - 1], poly[i]
            ia, ib = inside(a), inside(b)
            if ib:
                if not ia:
                    out.append(cut(a, b))
                out.append(b)
            elif ia:
                out.append(cut(a, b))
        return out

    def lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    poly = [tuple(p) for p in subject]
    for inside, cut in (
        (lambda p: p[0] >= x0, lambda a, b: lerp(a, b, (x0 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= x1, lambda a, b: lerp(a, b, (x1 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= y0, lambda a, b: lerp(a, b, (y0 - a[1]) / (b[1] - a[1]))),
        (lambda p: p[1] <= y1, lambda a, b: lerp(a, b, (y1 - a[1]) / (b[1] - a[1]))),
    ):
        poly = edge(poly, inside, cut)
        if not poly:
            return np.zeros((0, 2))
    return np.array(poly)


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography mapping 4 src points to 4 dst points."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    _, _, vt = np.linalg.svd(np.array(A, float))
    return vt[-1].reshape(3, 3) / vt[-1, -1]


def iou_doc_frame(pred: np.ndarray, true: np.ndarray, mw: float, mh: float) -> float:
    """SmartDoc Ch1 Jaccard index: both quads in the document frame (true quad -> mw x mh rectangle)."""
    rect = np.array([[0, 0], [mw, 0], [mw, mh], [0, mh]], float)
    Hm = homography(true, rect)
    ph = np.hstack([pred, np.ones((4, 1))]) @ Hm.T
    if np.any(ph[:, 2] <= 1e-9):  # a predicted corner maps behind the camera: no meaningful overlap
        return 0.0
    pt = ph[:, :2] / ph[:, 2:]
    inter = _area(_clip(pt, 0, 0, mw, mh)) if len(_clip(pt, 0, 0, mw, mh)) >= 3 else 0.0
    union = _area(pt) + mw * mh - inter
    return float(inter / union) if union > 0 else 0.0


def iou_img(pred: np.ndarray, true: np.ndarray, W: int, H: int) -> float:
    """IoU of two quads as pixel masks in the image plane (the prediction clipped to the frame)."""
    a, b = synth.poly_mask(pred, W, H) >= 0.5, synth.poly_mask(true, W, H) >= 0.5
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def eval_one(item: dict) -> dict:
    """Metrics for one item {"path", "row" (labels dict), "scale" (stored px -> original px), "kind"}."""
    with Image.open(item["path"]) as f:
        img = f.convert("RGB")
    W, H = img.size
    t0 = time.perf_counter()
    r = infer.detect(img, _MODEL)
    ms = (time.perf_counter() - t0) * 1000
    pred = np.asarray(r["corners"], float)
    true = _quad(item["row"])
    out = {"id": item["id"], "found": bool(r["page_found"]), "confidence": float(r["confidence"]),
           "negative": true is None, "ms": ms, "pred": pred.round(2).tolist()}
    if true is not None:
        s = item["scale"]
        e = match_err(pred, true) * s
        diag = float(np.hypot(W, H)) * s
        out.update({"iou_img": iou_img(pred, true, W, H), "err_px": e, "err_pct_img": 100 * e / diag})
        if item["kind"] == "smartdoc":
            out["iou_doc"] = iou_doc_frame(pred, true, item["row"]["model_width"], item["row"]["model_height"])
    return out


def _init(model: dict) -> None:
    global _MODEL
    _MODEL = model
    threadpool_limits(1)


def run(model: dict, items: list[dict], workers: int = 4) -> pd.DataFrame:
    """eval_one over items in parallel; one row per image."""
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(model,)) as pool:
        return pd.DataFrame(pool.map(eval_one, items, chunksize=8))


def doccorner_items(data_home: str, split: str, n: int | None = None) -> list[dict]:
    """Prepared DocCornerDataset items (first n of the seeded order when n is given)."""
    root = os.path.join(data_home, "prepared", "doccorner", split)
    lab = pd.read_csv(os.path.join(root, "labels.csv"))
    lab = lab.sample(frac=1.0, random_state=7).reset_index(drop=True)
    if n:
        lab = lab.head(n)
    return [{"id": r["filename"], "path": os.path.join(root, r["filename"]), "row": r, "kind": "doccorner",
             "scale": r["orig_width"] / r["width"]} for r in lab.to_dict("records")]


def smartdoc_items(data_home: str, manifest: str) -> list[dict]:
    """SmartDoc Ch1 test frames listed in the manifest (full 1920 x 1080 frames)."""
    lab = pd.read_csv(manifest)
    meta = pd.read_csv(os.path.join(data_home, "smartdoc_ch1_home", "frames", "metadata.csv.gz"),
                       usecols=["image_path", "model_width", "model_height"])
    lab = lab.merge(meta, on="image_path", how="left")
    fr = os.path.join(data_home, "smartdoc_ch1_home", "frames")
    return [{"id": r["image_path"], "path": os.path.join(fr, r["image_path"]), "row": r, "kind": "smartdoc",
             "scale": 1.0} for r in lab.to_dict("records")]


def summarise(df: pd.DataFrame, smartdoc: bool = False) -> dict:
    """Aggregate metrics: positives (IoU, corner error), negatives (false alarms), objectness P/R."""
    pos, neg = df[~df.negative], df[df.negative]
    out = {"n": int(len(df)), "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    if len(pos):
        out.update({
            "iou_img_mean": round(float(pos.iou_img.mean()), 4),
            "iou_img_ge_0.9": round(float((pos.iou_img >= 0.9).mean()), 4),
            "err_px_mean": round(float(pos.err_px.mean()), 1), "err_px_median": round(float(pos.err_px.median()), 1),
            "err_pct_img_mean": round(float(pos.err_pct_img.mean()), 3),
            "err_pct_img_median": round(float(pos.err_pct_img.median()), 3),
        })
        if smartdoc:
            out.update({"iou_doc_mean": round(float(pos.iou_doc.mean()), 4),
                        "iou_doc_ge_0.9": round(float((pos.iou_doc >= 0.9).mean()), 4)})
    tp = int(pos.found.sum()) if len(pos) else 0
    fp = int(neg.found.sum()) if len(neg) else 0
    out["objectness_recall"] = round(tp / len(pos), 4) if len(pos) else None
    out["objectness_precision"] = round(tp / (tp + fp), 4) if tp + fp else None
    out["false_alarm_rate_on_negatives"] = round(fp / len(neg), 4) if len(neg) else None
    return out

"""Scan a photo of a paper document: find the page, correct the perspective, enhance.

`scan(image, mode)` is the contract in docs/PLAN.md. It never raises on a valid PIL image: with no
model file, or when no plausible page is found, it returns the whole image enhanced (method "fallback").
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import joblib
import numpy as np
from PIL import Image, ImageFilter

from scanner import enhance as enh
from scanner import features as feats
from scanner import geometry as geo
from scanner.synth import poly_mask

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SYNTHETIC_MODEL_PATH = os.path.join(ROOT, "models", "scanner.joblib")  # trained on synthetic photos only
SYNTHETIC_METRICS_PATH = os.path.join(ROOT, "models", "scanner_metrics.json")
REAL_MODEL_PATH = os.path.join(ROOT, "models", "scanner_real.joblib")  # fine-tuned on real photos (docs/finetune)
REAL_METRICS_PATH = os.path.join(ROOT, "models", "scanner_real_metrics.json")
# The app uses the fine-tuned model when it is present, else the synthetic one.
_use_real = os.path.exists(REAL_MODEL_PATH)
MODEL_PATH = REAL_MODEL_PATH if _use_real else SYNTHETIC_MODEL_PATH
METRICS_PATH = REAL_METRICS_PATH if _use_real else SYNTHETIC_METRICS_PATH
REFINE_SIDE = 640  # px, long side of the image used for edge refinement
MIN_CONFIDENCE = 0.35
MIN_AREA_FRAC = 0.05  # of the photo
WORK_SIDE = 2400  # px, photos are downscaled to at most this long side before warping
_lock = threading.Lock()


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict | None:
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_model(path: str | None = None) -> dict | None:
    """The trained model bundle, cached; None when the file is missing or unreadable."""
    path = MODEL_PATH if path is None else path
    if not os.path.exists(path):
        return None
    with _lock:
        return _load(path, os.path.getmtime(path))


def predict_prob(img: Image.Image, model: dict) -> np.ndarray:
    """Page probability per grid cell, (gh, gw) float."""
    X, (gw, gh) = feats.compute(img, model["grid"])
    p = model["stage1"].predict_proba(X)[:, 1].reshape(gh, gw)
    if model.get("stage2") is not None:
        p = model["stage2"].predict_proba(np.hstack([X, feats.context(p)]))[:, 1].reshape(gh, gw)
    return p


def otsu_prob(img: Image.Image, grid: int = feats.GRID) -> np.ndarray:
    """Non-ML baseline: 1 where grid brightness is above the Otsu threshold, else 0."""
    gw, gh = feats.grid_shape(*img.size, grid)
    lum = np.asarray(img.convert("L").resize((gw, gh), Image.BOX), np.float64)
    hist = np.bincount(lum.astype(np.uint8).ravel(), minlength=256).astype(float)
    w0 = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    w1 = w0[-1] - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w0 - mu * w0[-1]) ** 2 / (w0 * w1)
    t = int(np.nanargmax(np.where(np.isfinite(between), between, np.nan)))
    return (lum > t).astype(np.float64)


def _refine_image(img: Image.Image) -> tuple[np.ndarray, float]:
    s = min(1.0, REFINE_SIDE / max(img.size))
    small = img.resize((max(8, int(round(img.size[0] * s))), max(8, int(round(img.size[1] * s)))), Image.BOX,
                       reducing_gap=3.0) if s < 1 else img
    a = np.asarray(small.filter(ImageFilter.GaussianBlur(1.0)), np.float32) / 255.0
    return a, small.size[0] / img.size[0]


def detect_from_prob(img: Image.Image, prob: np.ndarray, refine: bool = True) -> dict:
    """Quad in photo px, confidence and page_found from a grid probability map."""
    W, H = img.size
    gh, gw = prob.shape
    out = {"page_found": False, "corners": np.array([[0, 0], [W, 0], [W, H], [0, H]], float), "confidence": 0.0}
    mask = geo.clean_mask(prob)
    qg = geo.fit_quad(mask, prob)
    if qg is None:
        return out
    q = qg * np.array([W / gw, H / gh])
    if refine:
        a, s = _refine_image(img)
        search = 1.5 * max(a.shape[:2]) / max(gw, gh)
        q = geo.refine_edges(q * s, a, search) / s
    # confidence: classifier agreement with the quad, times shape plausibility
    inside = feats_poly(q * np.array([gw / W, gh / H]), gw, gh)
    pin = float(prob[inside].mean()) if inside.any() else 0.0
    pout = float(prob[~inside].mean()) if (~inside).any() else 0.0
    shape = float(np.clip((geo.angles(q).min() - 35) / 25, 0, 1))
    area = geo._area(q) / (W * H)
    conf = pin * (1 - pout) * shape
    out.update(corners=q, confidence=float(np.clip(conf, 0, 1)),
               page_found=bool(conf >= MIN_CONFIDENCE and area >= MIN_AREA_FRAC and geo.is_convex(q)))
    return out


def feats_poly(qg: np.ndarray, gw: int, gh: int) -> np.ndarray:
    """(gh, gw) bool: grid cells whose centre is inside the quad (grid coordinates)."""
    return poly_mask(qg, gw, gh) >= 0.5


def detect(img: Image.Image, model: dict | None = None) -> dict:
    """Detect the page with the ML model (quad in photo px). Raises if model is None."""
    model = load_model() if model is None else model
    prob = predict_prob(img, model)
    res = detect_from_prob(img, prob, model.get("refine", True))
    res["prob"] = prob
    return res


def _fallback(img: Image.Image, mode: str, version: str, conf: float = 0.0) -> dict:
    W, H = img.size
    try:
        page = enh.enhance(img, mode)
    except Exception:
        page = img
    return {"page_found": False, "corners": [[0.0, 0.0], [float(W), 0.0], [float(W), float(H)], [0.0, float(H)]],
            "confidence": float(conf), "image": page, "model_version": version, "method": "fallback"}


def scan(image: Image.Image, mode: str = "gray") -> dict:
    """Find the page, correct perspective, enhance. mode: "color" | "gray" | "bw".

    Returns {"page_found", "corners" (TL TR BR BL, source px), "confidence" 0..1, "image" (PIL),
    "model_version", "method" ("ml" | "fallback")}.
    """
    if mode not in enh.MODES:
        raise ValueError(f"mode must be one of {enh.MODES}")
    img = image.convert("RGB") if image.mode != "RGB" else image
    W, H = img.size
    model = load_model()
    version = model.get("model_version", "unknown") if model else "none"
    if model is None:
        return _fallback(img, mode, version)
    try:
        s = min(1.0, WORK_SIDE / max(W, H))
        work = img.resize((max(1, int(W * s)), max(1, int(H * s))), Image.BOX, reducing_gap=3.0) if s < 1 else img
        res = detect(work, model)
        if not res["page_found"]:
            return _fallback(work if s < 1 else img, mode, version, res["confidence"]) | {
                "corners": [[0.0, 0.0], [float(W), 0.0], [float(W), float(H)], [0.0, float(H)]]}
        q = res["corners"]
        page = enh.enhance(geo.warp(work, q), mode)
        return {"page_found": True, "corners": (q / s).round(1).tolist(), "confidence": round(res["confidence"], 3),
                "image": page, "model_version": version, "method": "ml"}
    except Exception:
        return _fallback(img, mode, version)


__all__ = ["scan", "detect", "detect_from_prob", "load_model", "otsu_prob", "predict_prob"]

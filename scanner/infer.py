"""Scan a photo of a paper document: find the page, correct the perspective, enhance.

`scan(image, mode)` is the contract in docs/PLAN.md. It never raises on a valid PIL image: with no
model file, or when no plausible page is found, it returns the whole image enhanced (method "fallback").
"""
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache

import joblib
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

from scanner import enhance as enh
from scanner.convert import to_rgb
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
# Chosen on the DocCornerDataset validation photos (models/finetune/eval/ft_real_only__doccorner_val.csv.gz)
# as the highest value that keeps at least 99.5 % of the well-outlined pages (IoU >= 0.9): there it cuts
# false alarms on photos without a document from 37 % (at the earlier 0.35) to 4 %. See docs/scanner.md.
MIN_CONFIDENCE = 0.60
MIN_AREA_FRAC = 0.05  # of the photo
WORK_SIDE = 2400  # px, photos are downscaled to at most this long side before warping
MIN_SIDE = 32  # px; a photo with a shorter side than this is never a page
# Noise guard (see coherence()): a quad is rejected only when ALL three hold, i.e. the inside and the
# outside of the quad do not differ in colour, nor in fine texture, and no side has a coherent step.
# Uniform random-noise photos gave d' <= 0.023, texture ratio within 1.5 % and step z <= 2.7; every
# synthetic photo (train/val/test/ood, 700) with a quad has d' >= 0.068 (docs/scanner.md).
NOISE_MAX_DPRIME = 0.05
NOISE_MAX_LOG_TEXTURE_RATIO = 0.1
NOISE_MAX_STEP_Z = 4.0
COHERENCE_SIDE = 320  # px, long side of the image the noise guard looks at
_lock = threading.Lock()
log = logging.getLogger("healthdiary.scanner")


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict | None:
    try:
        return joblib.load(path)
    except Exception as err:  # missing, damaged, or written by another scikit-learn version
        if os.path.exists(path):
            log.warning("Scanner model %s could not be loaded (%s: %s).", path, type(err).__name__, err)
        return None


def load_model(path: str | None = None) -> dict | None:
    """The trained model bundle, cached; None when the file is missing or unreadable."""
    default = path is None
    path = MODEL_PATH if default else path
    if not os.path.exists(path):
        return None
    with _lock:
        model = _load(path, os.path.getmtime(path))
        # An unreadable fine-tuned model must not switch scanning off: use the synthetic model instead.
        if model is None and default and path != SYNTHETIC_MODEL_PATH and os.path.exists(SYNTHETIC_MODEL_PATH):
            model = _load(SYNTHETIC_MODEL_PATH, os.path.getmtime(SYNTHETIC_MODEL_PATH))
        return model


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


def coherence(img: Image.Image, q: np.ndarray) -> dict:
    """How clearly quad q (photo px) separates a page from its surroundings.

    "dprime": distance of mean colour inside vs outside over the pooled standard deviation;
    "texture_ratio": mean gradient magnitude inside over outside; "step_z": the largest, over the
    4 sides, of |mean colour step across the side| over its standard error along the side (a real
    page edge is a step that holds all along the side). None where it cannot be measured (the quad
    fills the photo, or its sides lie outside it)."""
    W, H = img.size
    s = min(1.0, COHERENCE_SIDE / max(W, H))
    small = img.resize((max(8, round(W * s)), max(8, round(H * s))), Image.BOX, reducing_gap=3.0) if s < 1 else img
    a = np.asarray(small.convert("RGB"), np.float32) / 255.0
    h, w = a.shape[:2]
    qs = np.asarray(q, float) * np.array([small.size[0] / W, small.size[1] / H])
    c0 = qs.mean(0)
    offs = np.arange(1.5, 6.6, 1.0)
    zs = []
    for i in range(4):
        p0, p1 = qs[i], qs[(i + 1) % 4]
        L = float(np.linalg.norm(p1 - p0))
        d = (p1 - p0) / (L + 1e-9)
        n = np.array([-d[1], d[0]])
        if np.dot((p0 + p1) / 2 - c0, n) < 0:
            n = -n  # outward
        ts = np.linspace(0.1, 0.9, int(np.clip(L * 0.1, 6, 60)))
        base = p0[None] + ts[:, None] * (p1 - p0)[None]
        strips, oks = [], []
        for sign in (-1, 1):
            P = base[:, None, :] + sign * offs[None, :, None] * n[None, None, :]
            xs, ys = P[..., 0] - 0.5, P[..., 1] - 0.5
            oks.append(((xs > 0) & (xs < w - 1) & (ys > 0) & (ys < h - 1)).all(1))
            strips.append(np.stack([ndi.map_coordinates(a[..., c], [ys.ravel(), xs.ravel()], order=1)
                                    .reshape(xs.shape).mean(1) for c in range(3)], -1))
        ok = oks[0] & oks[1]
        if ok.sum() < 6:
            continue
        dj = (strips[1] - strips[0])[ok]
        m = dj.mean(0)
        se = np.sqrt(((dj - m) ** 2).sum(1).mean() / len(dj))
        zs.append(float(np.linalg.norm(m) / (se + 1e-6)))
    inside = poly_mask(qs, w, h) >= 0.5
    out = {"dprime": None, "texture_ratio": None, "step_z": max(zs) if zs else None}
    if inside.any() and (~inside).mean() > 0.02:
        lum = a.mean(2)
        tex = np.hypot(ndi.sobel(lum, 0), ndi.sobel(lum, 1))
        sd = np.sqrt((a[inside].var(0).mean() + a[~inside].var(0).mean()) / 2)
        out["dprime"] = float(np.linalg.norm(a[inside].mean(0) - a[~inside].mean(0)) / (sd + 1e-6))
        out["texture_ratio"] = float(tex[inside].mean() / (tex[~inside].mean() + 1e-6))
    return out


def looks_like_noise(c: dict) -> bool:
    """True when the inside and outside of the quad are indistinguishable (see NOISE_* above)."""
    if c["dprime"] is None or c["texture_ratio"] is None:
        return False
    return (c["dprime"] < NOISE_MAX_DPRIME
            and abs(np.log(max(c["texture_ratio"], 1e-6))) < NOISE_MAX_LOG_TEXTURE_RATIO
            and (c["step_z"] is None or c["step_z"] < NOISE_MAX_STEP_Z))


def detect_from_prob(img: Image.Image, prob: np.ndarray, refine: bool = True) -> dict:
    """Quad in photo px, confidence and page_found from a grid probability map."""
    W, H = img.size
    gh, gw = prob.shape
    out = {"page_found": False, "corners": np.array([[0, 0], [W, 0], [W, H], [0, H]], float), "confidence": 0.0}
    if min(W, H) < MIN_SIDE:  # far too small to hold a page; the classifier's output means nothing here
        return out
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
    found = bool(conf >= MIN_CONFIDENCE and area >= MIN_AREA_FRAC and geo.is_convex(q))
    if found:
        coh = coherence(img, q)
        out["coherence"] = coh
        if looks_like_noise(coh):
            found = False
            out["rejected"] = "no coherent page edge: inside and outside look the same (noise)"
    out.update(corners=q, confidence=float(np.clip(conf, 0, 1)), page_found=found)
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
    img = to_rgb(image)  # alpha on white, 16-bit and float grey scaled (a plain convert loses them)
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


__all__ = ["scan", "detect", "detect_from_prob", "load_model", "otsu_prob", "predict_prob", "coherence",
           "looks_like_noise"]

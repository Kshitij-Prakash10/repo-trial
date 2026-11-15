"""Synthetic handwriting lines: text rendered with open-licence handwriting fonts, then degraded.

Fonts come from data/raw/fonts (SIL OFL, scripts/download_handwriting.py). Every image is synthetic
and is only ever used for training. Deterministic for a given numpy Generator.
"""
from __future__ import annotations

import glob
import io
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr.hwdata import RAW

_FONTS: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font_files() -> list[str]:
    return sorted(glob.glob(os.path.join(RAW, "fonts", "*.ttf")))


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    if (path, size) not in _FONTS:
        _FONTS[(path, size)] = ImageFont.truetype(path, size)
    return _FONTS[(path, size)]


def _elastic(img: np.ndarray, rng: np.random.Generator, alpha: float, sigma: float) -> np.ndarray:
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def render_line(text: str, rng: np.random.Generator, fonts: list[str] | None = None) -> np.ndarray:
    """One synthetic handwritten line (RGB uint8). Characters the font lacks are still drawn as boxes,
    so callers should pass plain ASCII text."""
    fonts = fonts or font_files()
    size = int(rng.integers(44, 72))
    font = _font(fonts[int(rng.integers(len(fonts)))], size)
    x0, y0, x1, y1 = font.getbbox(text)
    pad = int(size * 0.5)
    W, H = max(8, x1 - x0 + 2 * pad), max(8, y1 - y0 + 2 * pad)
    ink = Image.new("L", (W, H), 0)
    ImageDraw.Draw(ink).text((pad - x0, pad - y0), text, fill=255, font=font)
    a = np.asarray(ink)
    k = int(rng.integers(0, 3))  # ink thickness
    if k == 1:
        a = cv2.dilate(a, np.ones((2, 2), np.uint8))
    elif k == 2 and size > 54:
        a = cv2.erode(a, np.ones((2, 2), np.uint8))
    a = _elastic(a, rng, alpha=float(rng.uniform(0, size * 0.12)), sigma=float(rng.uniform(size * 0.25, size * 0.6)))
    shear, rot = float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-2.5, 2.5))  # slant and baseline tilt
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, 1.0)
    M[0, 1] += shear
    M[0, 2] -= shear * H / 2
    a = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)
    ys, xs = np.where(a > 40)
    if len(xs):
        m = int(size * rng.uniform(0.08, 0.3))
        a = a[max(0, ys.min() - m):ys.max() + m, max(0, xs.min() - m):xs.max() + m]
    h, w = a.shape
    paper = np.full((h, w, 3), rng.uniform(190, 255), np.float32) * rng.uniform(0.92, 1.0, 3)
    paper += cv2.GaussianBlur(rng.normal(0, 14, (h, w)).astype(np.float32), (0, 0), max(h, 8) / 3)[..., None]  # shading
    if rng.random() < 0.3:  # ruled paper
        y = int(h * rng.uniform(0.7, 0.9))
        paper[y:y + 2] *= rng.uniform(0.75, 0.9)
    colour = np.array([[20, 20, 30], [15, 25, 110], [40, 40, 45], [90, 20, 20]][int(rng.integers(4))], np.float32)
    colour = np.clip(colour + rng.normal(0, 10, 3), 0, 255)
    alpha = (a.astype(np.float32) / 255 * rng.uniform(0.7, 1.0))[..., None]
    img = paper * (1 - alpha) + colour * alpha
    img += rng.normal(0, rng.uniform(0, 6), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.3, 1.3)))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=int(rng.integers(35, 90)))
        img = np.asarray(Image.open(buf).convert("RGB"))
    return img

"""Page enhancement: shading removal, contrast stretch, and adaptive threshold for black-and-white."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

MODES = ("color", "gray", "bw")


def _background(ch: Image.Image) -> np.ndarray:
    """Paper brightness estimate (float, same size): max filter to erase ink, then a large blur."""
    w, h = ch.size
    s = max(1, int(max(w, h) / 400))
    small = ch.resize((max(1, w // s), max(1, h // s)), Image.BOX)
    k = max(3, (int(max(small.size) / 60) // 2) * 2 + 1)
    bg = small.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.GaussianBlur(max(small.size) / 30))
    return np.asarray(bg.resize((w, h), Image.BILINEAR), np.float32)


def _flatten(ch: Image.Image) -> np.ndarray:
    """Shading-corrected channel in 0..1 (paper near 1)."""
    a = np.asarray(ch, np.float32)
    return np.clip(a / np.maximum(_background(ch), 8.0), 0, 1.2)


def _stretch(a: np.ndarray, lo_p: float = 1.0, hi_p: float = 95.0) -> np.ndarray:
    lo, hi = np.percentile(a, lo_p), np.percentile(a, hi_p)
    if hi - lo < 1e-3:
        return np.clip(a, 0, 1)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def enhance(img: Image.Image, mode: str = "gray") -> Image.Image:
    """Enhance a page photo. mode: color (RGB), gray (L) or bw (L, values 0 or 255)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    img = img.convert("RGB")
    if mode == "color":
        chans = [_flatten(c) for c in img.split()]
        lum = 0.299 * chans[0] + 0.587 * chans[1] + 0.114 * chans[2]
        lo, hi = np.percentile(lum, 1.0), np.percentile(lum, 95.0)
        sc = max(hi - lo, 1e-3)
        out = np.stack([np.clip((c - lo) / sc, 0, 1) for c in chans], -1)
        return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGB")
    g = _stretch(_flatten(img.convert("L")))
    if mode == "gray":
        return Image.fromarray((g * 255 + 0.5).astype(np.uint8), "L")
    gi = Image.fromarray((g * 255 + 0.5).astype(np.uint8), "L")
    gs = np.asarray(gi.filter(ImageFilter.GaussianBlur(max(0.8, max(img.size) / 1500))), np.float32) / 255.0  # denoise
    r = max(2.0, max(img.size) / 80)
    local = np.asarray(gi.filter(ImageFilter.BoxBlur(r)), np.float32) / 255.0
    ink = ((gs < local - 0.08) & (gs < 0.85)) | (gs < 0.35)
    return Image.fromarray(np.where(ink, 0, 255).astype(np.uint8), "L")

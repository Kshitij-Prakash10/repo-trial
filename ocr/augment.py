"""Augmentations that mimic phone captures: perspective, blur, shadow, low contrast, JPEG artefacts, noise.

All functions take and return uint8 RGB arrays. Every random choice comes from the numpy Generator
passed in, so a seed reproduces the stream.
"""
from __future__ import annotations

import cv2
import numpy as np


def perspective(img: np.ndarray, rng: np.random.Generator, strength: float, boxes: np.ndarray | None = None):
    """Move each corner by up to strength x the image size. Returns (image, boxes); boxes are [x0,y0,x1,y1]
    and become the bounding rectangles of the warped boxes."""
    h, w = img.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    dst = (src + rng.uniform(-strength, strength, (4, 2)) * [w, h]).astype(np.float32)
    dst -= dst.min(0)
    ow, oh = int(np.ceil(dst[:, 0].max())), int(np.ceil(dst[:, 1].max()))
    M = cv2.getPerspectiveTransform(src, dst)
    fill = [int(v) for v in np.median(img.reshape(-1, 3), axis=0)]
    out = cv2.warpPerspective(img, M, (max(ow, 2), max(oh, 2)), flags=cv2.INTER_LINEAR, borderValue=fill)
    if boxes is None or len(boxes) == 0:
        return out, boxes
    b = np.asarray(boxes, np.float32)
    corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], 1).reshape(-1, 1, 2)
    t = cv2.perspectiveTransform(corners, M).reshape(-1, 4, 2)
    return out, np.concatenate([t.min(1), t.max(1)], 1)


def shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A soft dark band across the image (a hand or phone shadow)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    d = (xx - rng.uniform(0, w)) * np.cos(ang) + (yy - rng.uniform(0, h)) * np.sin(ang)
    soft = max(4.0, rng.uniform(0.05, 0.4) * max(h, w))
    mask = 1.0 / (1.0 + np.exp(-d / soft))
    lo = rng.uniform(0.45, 0.85)
    return np.clip(img * (lo + (1 - lo) * mask)[..., None], 0, 255).astype(np.uint8)


def low_contrast(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a, shift = rng.uniform(0.45, 0.9), rng.uniform(-25, 25)
    m = img.mean()
    return np.clip((img - m) * a + m + shift, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, rng: np.random.Generator, max_sigma: float) -> np.ndarray:
    if rng.random() < 0.5:  # defocus
        return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, max_sigma))
    k = int(rng.integers(2, max(3, int(3 * max_sigma)) + 1))  # motion
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))


def jpeg(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 80))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy() if ok else img


def noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(img + rng.normal(0, rng.uniform(2, 10), img.shape), 0, 255).astype(np.uint8)


def phone_capture(img: np.ndarray, rng: np.random.Generator, boxes: np.ndarray | None = None,
                  p: float = 0.5, warp: float = 0.04, max_sigma: float = 1.5):
    """Apply each effect with probability p (perspective first, JPEG last). Returns (image, boxes)."""
    if rng.random() < p:
        img, boxes = perspective(img, rng, warp, boxes)
    if rng.random() < p:
        img = shadow(img, rng)
    if rng.random() < p:
        img = low_contrast(img, rng)
    if rng.random() < p:
        img = blur(img, rng, max_sigma)
    if rng.random() < p * 0.6:
        img = noise(img, rng)
    if rng.random() < p:
        img = jpeg(img, rng)
    return img, boxes

"""Per-pixel features on a downscaled grid for the page segmenter.

The photo is box-downscaled so that its long side is GRID cells. Each cell gets colour, image-normalised
colour, local statistics at several box-filter scales, gradient and fine texture, and position features.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

GRID = 128  # cells on the long side
FINE = 4  # the texture feature uses a FINE x finer grid


def grid_shape(W: int, H: int, grid: int = GRID) -> tuple[int, int]:
    """(gw, gh) grid size for a W x H photo, long side = grid."""
    s = grid / max(W, H)
    return max(8, int(round(W * s))), max(8, int(round(H * s)))


def _resize(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(img.resize(size, Image.BOX, reducing_gap=3.0), np.float32) / 255.0


def _sobel(a: np.ndarray) -> np.ndarray:
    return np.hypot(ndi.sobel(a, 0, mode="nearest"), ndi.sobel(a, 1, mode="nearest"))


FEATURE_NAMES: list[str] = []


def compute(img: Image.Image, grid: int = GRID) -> tuple[np.ndarray, tuple[int, int]]:
    """Features for every grid cell: (gh * gw, n_features) float32, and (gw, gh)."""
    img = img.convert("RGB")
    gw, gh = grid_shape(*img.size, grid)
    rgb = _resize(img, (gw, gh))
    fine = _resize(img, (gw * FINE, gh * FINE)).mean(2)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    mx, mn = rgb.max(2), rgb.min(2)
    sat = (mx - mn) / (mx + 1e-3)
    opp_a, opp_b = R - G, (R + G) / 2 - B
    f: dict[str, np.ndarray] = {"r": R, "g": G, "b": B, "lum": lum, "sat": sat, "chroma": mx - mn,
                                "opp_a": opp_a, "opp_b": opp_b}
    # image-normalised colour: background statistics from the border ring, page prior from the centre
    ring = np.zeros((gh, gw), bool)
    ring[:2], ring[-2:], ring[:, :2], ring[:, -2:] = True, True, True, True
    cen = np.zeros((gh, gw), bool)
    cen[int(gh * 0.35):int(gh * 0.65) + 1, int(gw * 0.35):int(gw * 0.65) + 1] = True
    lab = np.stack([lum, opp_a, opp_b], -1)
    bmed = np.median(lab[ring], 0)
    cmed = np.median(lab[cen], 0)
    f["d_border"] = np.linalg.norm(lab - bmed, axis=-1)
    f["d_centre"] = np.linalg.norm(lab - cmed, axis=-1)
    f["lum_minus_border"] = lum - bmed[0]
    f["lum_minus_centre"] = lum - cmed[0]
    f["lum_rank"] = (np.argsort(np.argsort(lum.ravel())) / lum.size).reshape(lum.shape).astype(np.float32)
    f["lum_over_max"] = lum / (np.percentile(lum, 99) + 1e-3)
    # illumination-invariant colour: chromaticity, and brightness relative to the local maximum
    tot = R + G + B + 1e-3
    f["rn"], f["gn"] = R / tot, G / tot
    chrom = np.stack([f["rn"], f["gn"], sat], -1)
    f["d_border_chroma"] = np.linalg.norm(chrom - np.median(chrom[ring], 0), axis=-1)
    f["d_centre_chroma"] = np.linalg.norm(chrom - np.median(chrom[cen], 0), axis=-1)
    sm = ndi.uniform_filter(lum, 3, mode="nearest")
    for s in (15, 31):
        f[f"lum_over_localmax{s}"] = lum / (ndi.maximum_filter(sm, s, mode="nearest") + 1e-3)
    for s in (5, 11, 25):
        f[f"lum_mean{s}"] = ndi.uniform_filter(lum, s, mode="nearest")
        f[f"dborder_mean{s}"] = ndi.uniform_filter(f["d_border"], s, mode="nearest")
    for s in (3, 7, 15):
        m = ndi.uniform_filter(lum, s, mode="nearest")
        f[f"lum_std{s}"] = np.sqrt(np.maximum(ndi.uniform_filter(lum * lum, s, mode="nearest") - m * m, 0))
    f["sat_mean7"] = ndi.uniform_filter(sat, 7, mode="nearest")
    grad = _sobel(ndi.gaussian_filter(lum, 0.7))
    f["grad"] = grad
    f["grad_mean5"] = ndi.uniform_filter(grad, 5, mode="nearest")
    tex = _sobel(fine).reshape(gh, FINE, gw, FINE).mean((1, 3))
    f["texture"] = tex
    f["texture_mean5"] = ndi.uniform_filter(tex, 5, mode="nearest")
    f["texture_mean13"] = ndi.uniform_filter(tex, 13, mode="nearest")
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    f["x"] = (xx + 0.5) / gw
    f["y"] = (yy + 0.5) / gh
    f["d_edge"] = np.minimum.reduce([xx + 0.5, yy + 0.5, gw - xx - 0.5, gh - yy - 0.5]) / max(gw, gh)
    f["d_centre_pos"] = np.hypot(f["x"] - 0.5, f["y"] - 0.5)
    if not FEATURE_NAMES:
        FEATURE_NAMES.extend(f)
    X = np.stack([f[k].astype(np.float32) for k in FEATURE_NAMES], -1).reshape(-1, len(FEATURE_NAMES))
    return X, (gw, gh)


def feature_names() -> list[str]:
    """Names of the columns returned by compute()."""
    if not FEATURE_NAMES:
        compute(Image.new("RGB", (32, 24), (128, 128, 128)))
    return list(FEATURE_NAMES)


def context(prob: np.ndarray) -> np.ndarray:
    """Auto-context features from a first-stage probability map (gh, gw) -> (gh * gw, 6)."""
    cols = [prob] + [ndi.uniform_filter(prob, s, mode="nearest") for s in (3, 7, 15, 31)]
    cols.append(ndi.maximum_filter(ndi.uniform_filter(prob, 5, mode="nearest"), 9, mode="nearest"))
    return np.stack(cols, -1).reshape(-1, len(cols)).astype(np.float32)

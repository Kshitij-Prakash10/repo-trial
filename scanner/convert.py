"""Turn any PIL image into 8-bit RGB without losing its content. Needs only Pillow and numpy.

A plain `image.convert("RGB")` makes transparent areas black, clamps 16-bit grey (I;16) to white and
cuts float images off at 0 and 255. `to_rgb` is used by the page scanner (scanner.infer.scan) and by
the OCR pipeline (ocr.pipeline.to_rgb is this function).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_ALPHA_MODES = ("RGBA", "LA", "PA", "RGBa", "La")
_DEEP_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")


def _grey_to_rgb(a: np.ndarray) -> Image.Image:
    g = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return Image.fromarray(g).convert("RGB")


# Alpha at and above which a pixel keeps its own colour (alpha_to_rgb). Below it the colour is blended towards
# white in proportion to the alpha, down to white at alpha 0.
ALPHA_FULL = 64


def has_alpha(image: Image.Image) -> bool:
    """Whether the image carries transparency: an alpha channel, or a transparent colour of an 8-bit image
    (palette, RGB, L, 1). A transparent value of 16-bit or deeper grey is ignored: it is scaled instead."""
    return image.mode in _ALPHA_MODES or ("transparency" in image.info and image.mode in ("P", "RGB", "L", "1"))


def alpha_to_rgb(image: Image.Image) -> Image.Image:
    """RGB copy of an image with transparency, decided pixel by pixel.

    out = white + (colour - white) * min(1, alpha / ALPHA_FULL). Fully transparent pixels become white:
    the colour stored under them means nothing (cut-out PNGs store black there). Pixels with alpha of at
    least ALPHA_FULL (a quarter) keep their own colour: compositing a photo's half see-through
    surroundings on white lightened them towards the page and hid its edge (20 synthetic test photos
    with alpha 128 outside the page: IoU >= 0.9 fell from 0.75 to 0.40; keeping the colour gives 0.75,
    docs/scanner.md). In between, the colour fades towards white, so nearly transparent pixels with
    meaningless dark colour do not turn black and the result changes little when alpha changes a little.

    Each pixel depends only on its own colour and alpha, so pixels elsewhere in the image never change
    it. (Until 2026-11 an image with no opaque pixel at all was composited on white instead, and adding
    one opaque pixel anywhere switched every half see-through pixel to its own colour.)"""
    a = np.asarray(image.convert("RGBA"))
    w = np.minimum(a[..., 3:].astype(np.float32) / ALPHA_FULL, 1.0)
    out = 255.0 + (a[..., :3].astype(np.float32) - 255.0) * w
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


def to_rgb(image: Image.Image) -> Image.Image:
    """RGB copy of any PIL image (the image itself when it is already RGB).

    Transparency (an alpha channel or a transparent colour, has_alpha) is handled by alpha_to_rgb, as in
    scanner.api.to_rgb8: transparent areas become white, pixels at least a quarter opaque keep their
    colour, fainter ones fade towards white. 16-bit grey is scaled from 0..65535 to 0..255. Float (F) and
    32-bit integer (I) grey are scaled by their largest value when that is above 1.0 (F) or 255 (I);
    when they hold negative values, the lowest value becomes black, so nothing is cut off at zero.
    """
    mode = image.mode
    if has_alpha(image):
        return alpha_to_rgb(image)
    if mode == "RGB":
        return image
    if mode in _DEEP_MODES:
        a = np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if a.size == 0:
            return Image.new("RGB", image.size)
        if mode.startswith("I;16"):
            return _grey_to_rgb(a / 65535.0 * 255.0)
        lo = min(0.0, float(a.min()))  # negative values: shift, so the lowest value becomes black
        hi = float(a.max()) - lo
        if mode == "I" and lo == 0.0 and hi <= 255:
            scale = 255.0  # 8-bit values stored in a 32-bit image
        else:
            scale = max(hi, 1.0)
        return _grey_to_rgb((a - lo) / scale * 255.0)
    return image.convert("RGB")


__all__ = ["to_rgb", "alpha_to_rgb", "has_alpha", "ALPHA_FULL"]

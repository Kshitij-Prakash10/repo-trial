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


def alpha_to_rgb(image: Image.Image) -> Image.Image:
    """RGB copy of an image with transparency.

    Fully transparent pixels become white: the colour stored under them means nothing (cut-out PNGs
    store black there). Partly transparent pixels keep their own colour: compositing a photo's half
    see-through surroundings on white lightened them towards the page and hid its edge (20 synthetic
    test photos with alpha 128 outside the page: IoU >= 0.9 fell from 0.75 to 0.40; keeping the colour
    gives 0.75 again, docs/scanner.md). An image with no opaque pixel at all is a see-through layer,
    such as half see-through text, and is shown over white paper."""
    a = np.asarray(image.convert("RGBA"))
    alpha = a[..., 3]
    if not (alpha == 255).any():
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, Image.fromarray(a, "RGBA")).convert("RGB")
    rgb = a[..., :3].copy()
    rgb[alpha == 0] = 255
    return Image.fromarray(rgb, "RGB")


def to_rgb(image: Image.Image) -> Image.Image:
    """RGB copy of any PIL image (the image itself when it is already RGB).

    Transparent areas become white, partly transparent ones keep their colour (alpha_to_rgb). 16-bit grey is scaled from 0..65535 to 0..255. Float (F) and
    32-bit integer (I) grey are scaled by their largest value when that is above 1.0 (F) or 255 (I);
    when they hold negative values, the lowest value becomes black, so nothing is cut off at zero.
    """
    mode = image.mode
    if mode == "RGB":
        return image
    if mode in _ALPHA_MODES or (mode == "P" and "transparency" in image.info):
        return alpha_to_rgb(image)
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


__all__ = ["to_rgb", "alpha_to_rgb"]

"""Page orientation (0 / 90 / 180 / 270 degrees) and small-angle deskew.

Orientation: a MobileNetV3-small classifier on the whole page (docTR's pretrained page-orientation
model, fine-tuned here on rotated printed and handwriting pages). Class k means the page has been
turned k quarter-turns counter-clockwise from upright; `upright()` turns it back.

Deskew: a projection-profile search. The page is binarised and rotated by trial angles within
+/- MAX_SKEW degrees; text lines are horizontal when the row sums are most uneven.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from ocr.common import ROOT

ORIENT_ARCH = "mobilenet_v3_small_page_orientation"
ORIENT_CKPT = os.path.join(ROOT, "checkpoints", "orientation_v1", "orientation.pt")
SIZE = 512  # px, square input (the pretrained model's size)
MEAN, STD = (0.694, 0.695, 0.693), (0.299, 0.296, 0.301)  # docTR's normalisation for this model
DOCTR_CLASSES = [0, -90, 180, 90]  # docTR's output order, angles clockwise-positive: its -90 is our k=1
MAX_SKEW = 15.0  # degrees
MIN_SKEW = 0.5  # smaller estimated angles are left alone
SKEW_MIN_GAIN = 1.08  # the profile must be this much sharper than at 0 degrees to rotate
FLIP_WORDS = 48  # largest word boxes used by the 180-degree check
FLIP_MARGIN = 0.10  # flip when crops read this much more confidently upside down (chosen on val)


def load_orient(ckpt: str | None = ORIENT_CKPT):
    """The classifier with outputs reordered to k = 0..3 quarter-turns counter-clockwise."""
    from doctr.models import classification
    model = getattr(classification, ORIENT_ARCH)(pretrained=True)
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return model.eval()


def letterbox(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Resize keeping the aspect ratio and pad symmetrically with black to size x size."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    out = np.zeros((size, size, 3), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out


def to_tensor(imgs: list[np.ndarray]) -> torch.Tensor:
    x = np.stack(imgs).astype(np.float32) / 255.0
    x = (x - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def logits_k(model, x: torch.Tensor) -> torch.Tensor:
    """Model outputs reordered from docTR's class order to k = 0, 1, 2, 3 (0, 90, 180, 270 CCW)."""
    out = model(x)
    order = [DOCTR_CLASSES.index(a) for a in (0, -90, 180, 90)]  # docTR counts clockwise
    return out[:, order]


@torch.no_grad()
def predict_k(model, img: np.ndarray) -> tuple[int, float]:
    """(k, confidence): the page was turned k quarter-turns counter-clockwise from upright."""
    p = torch.softmax(logits_k(model, to_tensor([letterbox(img)])), 1)[0]
    k = int(p.argmax())
    return k, float(p[k])


def upright(img: np.ndarray, k: int) -> np.ndarray:
    """Undo k counter-clockwise quarter-turns."""
    return np.ascontiguousarray(np.rot90(img, -k)) if k % 4 else img


def _profile_score(binary: np.ndarray, angle: float) -> float:
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rows = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST).sum(axis=1).astype(np.float64)
    return float(((rows[1:] - rows[:-1]) ** 2).sum())


def estimate_skew(img: np.ndarray) -> tuple[float, float]:
    """(angle in degrees, gain). Rotating the page by `angle` (counter-clockwise positive, as
    cv2.getRotationMatrix2D) makes its text lines horizontal; gain is how much sharper the row
    profile is at that angle than at 0."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    s = 700 / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    zero = _profile_score(b, 0.0)
    best, best_s = 0.0, zero
    for step, span in ((1.0, MAX_SKEW), (0.25, 1.0)):
        for a in np.arange(best - span, best + span + 1e-6, step):
            if abs(a) > MAX_SKEW:
                continue
            sc = _profile_score(b, float(a))
            if sc > best_s:
                best, best_s = float(a), sc
    return best, best_s / max(zero, 1e-9)


def deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    """(page, angle applied). The page is rotated only when the estimate is clear and not tiny."""
    angle, gain = estimate_skew(img)
    if abs(angle) < MIN_SKEW or gain < SKEW_MIN_GAIN:
        return img, 0.0
    return rotate(img, angle), angle


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre by angle degrees (counter-clockwise), growing the canvas; edges replicated."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def flip_margin(rec, img: np.ndarray, boxes: np.ndarray) -> tuple[float, int]:
    """(mean recogniser confidence as-is minus turned 180 degrees, number of crops used).

    The classifier never confuses the 90-degree axis on validation; all its errors are 180-degree
    flips. A word recogniser reads upright crops far more confidently than upside-down ones, so a
    clearly negative value means the page is still upside down. rec is a docTR recognition predictor.
    """
    if not len(boxes):
        return 0.0, 0
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sel = boxes[np.argsort(-area)[:FLIP_WORDS]]
    crops = [img[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] for x0, y0, x1, y1 in sel]
    c0 = np.mean([c for _, c in rec(crops)])
    c2 = np.mean([c for _, c in rec([np.ascontiguousarray(c[::-1, ::-1]) for c in crops])])
    return float(c0 - c2), len(crops)

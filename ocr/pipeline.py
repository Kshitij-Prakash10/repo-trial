"""The OCR pipeline: scanner stage -> text detection -> text recognition -> ordered text.

    from ocr.pipeline import OcrPipeline
    result = OcrPipeline.load().run(Image.open("photo.jpg"))
    result["text"]; result["words"]; result["lines"]

Stage 1 is the existing page scanner (scanner.infer.detect + scanner.geometry.warp, weights unchanged).
It was trained on photos of a page lying on a surface, and its quad is not always exact, so its crop
is only used when it is safe (`scanner_decision`): the page must be found with enough confidence, and
the quad must contain the text that the detector finds on the uncropped photo. Otherwise the whole
photo goes on to OCR. This keeps scans and screenshots (no page border) from being cropped wrongly.
"""
from __future__ import annotations

import os
import time

import numpy as np
from PIL import Image

from ocr import models
from ocr.common import group_lines, union
from scanner import geometry as geo
from scanner import infer as scanner_infer

MAX_SIDE = 2400  # px; larger photos are reduced first (the scanner's own working size)
SCAN_MIN_CONF = 0.5  # scanner confidence needed to use its crop
SCAN_MIN_TEXT_INSIDE = 0.98  # share of detected word centres that must lie inside the page quad
SCAN_MAX_AREA = 0.9  # a quad covering more of the photo than this changes nothing useful; skip the warp
MIN_BOX = 3  # px


def _inside(points: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Which points lie inside a convex quad (TL TR BR BL)."""
    ok = np.ones(len(points), bool)
    sign = None
    for a, b in zip(quad, np.roll(quad, -1, 0)):
        cross = (b[0] - a[0]) * (points[:, 1] - a[1]) - (b[1] - a[1]) * (points[:, 0] - a[0])
        s = cross >= 0
        sign = s if sign is None else sign
        ok &= s == sign
    return ok


def scanner_decision(page: dict, boxes: np.ndarray, W: int, H: int) -> tuple[bool, str]:
    """Use the scanner's crop? (decision, reason). page is scanner.infer.detect's result."""
    if not page["page_found"]:
        return False, "no page found"
    if page["confidence"] < SCAN_MIN_CONF:
        return False, f"confidence {page['confidence']:.2f} below {SCAN_MIN_CONF}"
    q = np.asarray(page["corners"], float)
    if geo._area(q) / (W * H) > SCAN_MAX_AREA:
        return False, "page fills the photo"
    if len(boxes) == 0:
        return False, "no text detected to check the crop against"
    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    share = float(_inside(centres, q).mean())
    if share < SCAN_MIN_TEXT_INSIDE:
        return False, f"crop would lose text ({share:.0%} of words inside)"
    return True, "page found and all text inside"


class OcrPipeline:
    """Holds the three models. `run` is safe to call repeatedly."""

    def __init__(self, det_model, rec_model, use_scanner: bool = True, name: dict | str = "custom"):
        self.det = models.det_predictor(det_model)
        self.rec = models.rec_predictor(rec_model)
        self.use_scanner = use_scanner
        self.scanner_model = scanner_infer.load_model() if use_scanner else None
        self.name = name

    @classmethod
    def load(cls, weights: str = "finetuned", use_scanner: bool = True) -> "OcrPipeline":
        """weights: "finetuned" (models/ocr/*.pt) or "pretrained" (docTR weights, no fine-tuning)."""
        if weights not in ("finetuned", "pretrained"):
            raise ValueError("weights must be 'finetuned' or 'pretrained'")
        ft = weights == "finetuned"
        # a stage with no fine-tuned checkpoint in models/ocr keeps its pretrained weights
        det = models.DET_CKPT if ft and os.path.exists(models.DET_CKPT) else None
        rec = models.REC_CKPT if ft and os.path.exists(models.REC_CKPT) else None
        if ft and rec is None and det is None:
            raise FileNotFoundError("no fine-tuned checkpoint in models/ocr; use weights='pretrained'")
        name = {"detector": "finetuned" if det else "pretrained", "recogniser": "finetuned" if rec else "pretrained"}
        return cls(models.load_det(ckpt=det), models.load_rec(ckpt=rec), use_scanner, name)

    def detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Word boxes (N, 4) in pixels of img and their detection scores."""
        out = self.det([img])[0]["words"]
        H, W = img.shape[:2]
        boxes = np.clip(out[:, :4] * np.array([W, H, W, H]), 0, [W, H, W, H])
        keep = (boxes[:, 2] - boxes[:, 0] >= MIN_BOX) & (boxes[:, 3] - boxes[:, 1] >= MIN_BOX)
        return boxes[keep], out[keep, 4]

    def recognise(self, img: np.ndarray, boxes: np.ndarray) -> list[tuple[str, float]]:
        crops = [img[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] for x0, y0, x1, y1 in boxes]
        return [(t, float(c)) for t, c in self.rec(crops)] if crops else []

    def run(self, image: Image.Image, keep_first_pass: bool = False) -> dict:
        """Full pipeline on one photo or scan. Returns the result dict described in README.md.

        keep_first_pass adds "first_pass_boxes": the detector's word boxes on the uncropped photo, in
        source pixels (used by the evaluation to score detection without a second run)."""
        t0 = time.perf_counter()
        img = image.convert("RGB")
        W0, H0 = img.size
        s = min(1.0, MAX_SIDE / max(W0, H0))
        if s < 1:
            img = img.resize((max(1, int(W0 * s)), max(1, int(H0 * s))), Image.BOX, reducing_gap=3.0)
        W, H = img.size
        arr = np.asarray(img)
        boxes, scores = self.detect(arr)
        first_pass = (boxes / s).round(1).tolist()
        scan = {"used": False, "reason": "scanner stage disabled", "page_found": None, "confidence": None,
                "corners": None}
        to_source = None  # maps page pixels back to the pixels of the working photo
        if self.use_scanner and self.scanner_model is not None:
            page = scanner_infer.detect(img, self.scanner_model)
            used, reason = scanner_decision(page, boxes, W, H)
            scan = {"used": used, "reason": reason, "page_found": bool(page["page_found"]),
                    "confidence": round(float(page["confidence"]), 3),
                    "corners": (np.asarray(page["corners"], float) / s).round(1).tolist()}
            if used:
                q = np.asarray(page["corners"], float)
                flat = geo.warp(img, q)
                pw, ph = flat.size
                to_source = geo.homography(np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], float), q)
                arr = np.asarray(flat)
                boxes, scores = self.detect(arr)
        elif self.use_scanner:
            scan["reason"] = "scanner model file missing"
        rec = self.recognise(arr, boxes)
        words = []
        for (x0, y0, x1, y1), sc, (text, conf) in zip(boxes, scores, rec):
            quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
            if to_source is not None:
                hq = np.hstack([quad, np.ones((4, 1))]) @ to_source.T
                quad = hq[:, :2] / hq[:, 2:]
            words.append({"text": text, "confidence": round(conf, 4), "detection_score": round(float(sc), 4),
                          "box": [round(float(v), 1) for v in (x0, y0, x1, y1)],
                          "source_quad": (quad / s).round(1).tolist()})
        lines = []
        for ln in group_lines(boxes):
            for w in ln:
                words[w]["line"] = len(lines)
            lines.append({"text": " ".join(words[w]["text"] for w in ln), "words": ln,
                          "box": [round(v, 1) for v in union(boxes[ln])],
                          "confidence": round(float(np.mean([words[w]["confidence"] for w in ln])), 4)})
        extra = {"first_pass_boxes": first_pass} if keep_first_pass else {}
        return {**extra, "text": "\n".join(ln["text"] for ln in lines), "lines": lines, "words": words,
                "page": {"width": int(arr.shape[1]), "height": int(arr.shape[0]),
                         "is_scanner_crop": bool(scan["used"])},
                "source": {"width": W0, "height": H0}, "scanner": scan, "models": self.name,
                "seconds": round(time.perf_counter() - t0, 3)}

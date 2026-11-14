"""The full scanner: orientation -> deskew -> page scanner -> detection -> routing -> recognition.

    from ocr.scan import scan
    text, confidences = scan("photo.jpg")

`scan` returns the page text and one confidence (0 to 1) per text line. `Scanner.run` returns the
full result: every line with its box, script (printed or handwritten), confidence and a
low-confidence flag, plus what the orientation and deskew steps did.

Printed lines are read word by word by the frozen printed recogniser (ocr.pipeline, unchanged).
Lines whose words look handwritten are cut out as straightened rectangles and read by the
handwriting recogniser (ocr.handwriting).
"""
from __future__ import annotations

import os
import time

import numpy as np
from PIL import Image, ImageOps

from ocr import orientation as O
from ocr import scriptclf
from ocr.handwriting import HandwritingReader
from ocr.hwdata import rect_crop
from ocr.pipeline import OcrPipeline, to_rgb

LOW_CONFIDENCE = 0.5  # lines below this are flagged
ORIENT_MIN_CONF = 0.6  # a turn is applied only when the classifier is at least this sure
HW_LINE_SHARE = 0.5  # a line is handwritten when its words' mean P(handwritten) reaches this
_DEFAULT: "Scanner | None" = None


class Scanner:
    """Holds all models. Stages can be switched off for ablations."""

    def __init__(self, orientation: bool = True, deskew: bool = True, handwriting: bool = True,
                 page_scanner: bool = True, postcorrect: bool = False, weights: str = "finetuned"):
        self.ocr = OcrPipeline.load(weights, use_scanner=page_scanner)
        self.orient = O.load_orient() if orientation else None
        self.do_deskew = deskew
        self.clf = scriptclf.load_clf() if handwriting else None
        self.hw = HandwritingReader() if handwriting else None
        self.postcorrect = postcorrect

    def run(self, image: Image.Image) -> dict:
        t0 = time.perf_counter()
        img = np.asarray(to_rgb(image))
        pre = {"quarter_turns": 0, "orientation_confidence": None, "deskew_degrees": 0.0}
        if self.orient is not None:
            k, conf = O.predict_k(self.orient, img)
            pre["orientation_confidence"] = round(conf, 4)
            if k and conf >= ORIENT_MIN_CONF:
                img, pre["quarter_turns"] = O.upright(img, k), k
        if self.do_deskew:
            img, angle = O.deskew(img)
            pre["deskew_degrees"] = round(angle, 2)
        base = self.ocr.run(Image.fromarray(img), keep_page=True)
        page = base.pop("page_image")
        words = base["words"]
        boxes = np.array([w["box"] for w in words], float).reshape(-1, 4)
        p_hw = np.zeros(len(words))
        if self.clf is not None and len(words):
            crops = [page[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] for x0, y0, x1, y1 in boxes]
            p_hw = scriptclf.handwritten_prob(self.clf, crops)
        lines, hw_idx, hw_crops = [], [], []
        for ln in base["lines"]:
            share = float(np.mean(p_hw[ln["words"]])) if ln["words"] else 0.0
            entry = {"text": ln["text"], "box": ln["box"], "confidence": ln["confidence"], "script": "printed",
                     "handwritten_share": round(share, 3), "words": ln["words"]}
            if self.hw is not None and share >= HW_LINE_SHARE:
                b = boxes[ln["words"]]
                pts = np.concatenate([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]])
                hw_idx.append(len(lines))
                hw_crops.append(rect_crop(page, pts))
            lines.append(entry)
        if hw_crops:
            for i, (text, conf) in zip(hw_idx, self.hw.read(hw_crops)):
                if self.postcorrect:
                    from ocr.postcorrect import correct_line
                    text = correct_line(text)
                lines[i].update(text=text, confidence=round(conf, 4), script="handwritten")
        for ln in lines:
            ln["low_confidence"] = bool(ln["confidence"] < LOW_CONFIDENCE)
        lines = [ln for ln in lines if ln["text"].strip()]
        return {"text": "\n".join(ln["text"] for ln in lines), "confidences": [ln["confidence"] for ln in lines],
                "lines": lines, "words": words, "preprocessing": pre, "page": base["page"], "source": base["source"],
                "scanner": base["scanner"], "models": {**(base["models"] if isinstance(base["models"], dict) else {}),
                                                       "handwriting": getattr(self.hw, "name", None)},
                "seconds": round(time.perf_counter() - t0, 3)}


def scan(image_path: str) -> tuple[str, list[float]]:
    """Read a photo or scan. Returns (text, confidences): the page text, one line per text line, and
    the confidence of each of those lines (0 to 1). Loads the models on first use."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Scanner()
    with Image.open(image_path) as f:
        image = ImageOps.exif_transpose(f)
        image.load()
    r = _DEFAULT.run(image)
    return r["text"], r["confidences"]

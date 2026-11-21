"""The full scanner: orientation -> (optional deskew, off by default) -> page scanner -> detection -> routing -> recognition.

    from ocr.scan import scan
    text, confidences = scan("photo.jpg")

`scan` returns the page text and one confidence (0 to 1) per text line. `Scanner.run` returns the
full result: every line with its box, script (printed or handwritten), confidence, a
low-confidence flag, a confirm-numbers flag (a number or unit in a line that went to the
handwriting reader, or a unit without a number of its own: always to be checked by a person), a
has-numbers flag (the shown text holds a number or unit; quieter, for printed lines), plus what the
orientation and deskew steps did.

Printed lines are read word by word by the frozen printed recogniser (ocr.pipeline, unchanged).
Lines whose words look handwritten are cut out as straightened rectangles and read by the
handwriting recogniser (ocr.handwriting) too; its reading replaces the printed one only when its
confidence is higher. On validation this keeps printed pages at least as good as printed-only
reading, while still reading most handwriting with the handwriting model.
"""
from __future__ import annotations

import re
import time

import numpy as np
from PIL import Image, ImageOps

from ocr import orientation as O
from ocr import scriptclf
from ocr.common import has_misread_number, has_number_or_unit, line_confirm_numbers, number_lost  # noqa: F401
from ocr.handwriting import HandwritingReader
from ocr.hwdata import rect_crop
from ocr.pipeline import OcrPipeline, to_rgb

LOW_CONFIDENCE = 0.5  # lines below this are flagged
ORIENT_MIN_CONF = 0.6  # a turn is applied only when the classifier is at least this sure
HW_LINE_SHARE = 0.5  # a line is handwritten when its words' mean P(handwritten) reaches this
_RULE_TEXT = re.compile(r"[-_=.:~\u2013\u2014\u00b7]{2,}")  # a printed reading of dashes / dots / rules only
_LETTER = re.compile(r"[^\W\d_]")
_ALNUM = re.compile(r"[^\W_]")
_DEFAULT: "Scanner | None" = None


class Scanner:
    """Holds all models. Stages can be switched off for ablations."""

    def __init__(self, orientation: bool = True, deskew: bool = False, handwriting: bool = True,
                 page_scanner: bool = True, postcorrect: bool = False, weights: str = "finetuned",
                 flip_check: bool = True, mask_neighbours: bool = False, flag_misread_digits: bool = False):
        if postcorrect:
            # Withdrawn: on medical text it rewrites drug names and units (digoxin -> dioxin,
            # mmol -> mol), and the evaluation found no gain from it (reports/final_report.md).
            raise ValueError("Lexicon post-correction is switched off: it changes medicine names and units.")
        self.ocr = OcrPipeline.load(weights, use_scanner=page_scanner)
        self.orient = O.load_orient() if orientation else None
        self.do_deskew = deskew
        self.flip_check = flip_check
        self.clf = scriptclf.load_clf() if handwriting else None
        self.hw = HandwritingReader() if handwriting else None
        self.postcorrect = postcorrect
        # Paint out words of neighbouring lines inside a handwriting line's crop (ocr.hwdata.rect_crop).
        # Off by default: its effect on real handwriting (GNHK) has not been measured yet.
        self.mask_neighbours = mask_neighbours
        # Also flag lines whose digits look misread as letters ("Metformin SOO my"), in any of their readings. Off by default:
        # measured only on invented lines, not on real pages (README, "Measured on synthetic data only").
        self.flag_misread_digits = flag_misread_digits

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
        if self.orient is not None and self.flip_check:
            margin, n = O.flip_margin(self.ocr.rec, page, boxes)
            pre["flip_check"] = {"margin": round(margin, 4), "crops": n, "flipped": bool(margin < -O.FLIP_MARGIN)}
            if margin < -O.FLIP_MARGIN:  # still upside down: turn it and read again
                img = O.upright(img, 2)
                pre["quarter_turns"] = (pre["quarter_turns"] + 2) % 4
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
                quads = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], 1)  # (n, 4, 2)
                hw_idx.append(len(lines))
                mask = getattr(self, "mask_neighbours", False)
                hw_crops.append(rect_crop(page, quads.reshape(-1, 2), words=list(quads) if mask else None))
            lines.append(entry)
        if hw_crops:
            for i, crop, (text, conf) in zip(hw_idx, hw_crops, self.hw.read(hw_crops)):
                lines[i].update(handwriting_text=text, handwriting_confidence=round(conf, 4))
                keep, why = keep_printed(lines[i]["text"], text, crop)
                if keep:
                    lines[i]["handwriting_rejected"] = why
                elif conf > lines[i]["confidence"]:  # keep the reading the readers are surer of (chosen on val)
                    lines[i].update(printed_text=lines[i]["text"], printed_confidence=lines[i]["confidence"],
                                    text=text, confidence=round(conf, 4), script="handwritten")
        routed = self.clf is not None and self.hw is not None  # else "printed" only means "not checked"
        misread = getattr(self, "flag_misread_digits", False)
        for ln in lines:
            ln["low_confidence"] = bool(ln["confidence"] < LOW_CONFIDENCE)
            ln["confirm_numbers"] = confirm_numbers(ln, script_known=routed, flag_misread_digits=misread)
            ln["has_numbers"] = has_number_or_unit(ln["text"])
        lines = [ln for ln in lines if ln["text"].strip()]
        return {"text": "\n".join(ln["text"] for ln in lines), "confidences": [ln["confidence"] for ln in lines],
                "lines": lines, "words": words, "preprocessing": pre, "page": base["page"], "source": base["source"],
                "scanner": base["scanner"], "models": {**(base["models"] if isinstance(base["models"], dict) else {}),
                                                       "handwriting": getattr(self.hw, "name", None)},
                "seconds": round(time.perf_counter() - t0, 3)}


def confirm_numbers(ln: dict, script_known: bool | None = None, misread_digits: bool = False, *,
                    flag_misread_digits: bool | None = None) -> bool:
    """Must a person check this line's numbers against the page, whatever its confidence?

    The decision is ocr.common.line_confirm_numbers (torch-free; see there). flag_misread_digits (the
    Scanner option of the same name, off by default) also flags letters that look like misread digits
    before a unit ("Metformin SOO my") in any of the line's readings; misread_digits is its older name.
    """
    flag = misread_digits if flag_misread_digits is None else flag_misread_digits
    return line_confirm_numbers(ln, script_known=script_known, flag_misread_digits=flag)


def looks_like_rule(crop: np.ndarray) -> bool:
    """True when a line crop holds a horizontal rule or a row of dashes / dots: the ink lies in a
    thin band (at most a fifth of the crop height) yet runs along at least half of its width."""
    if crop is None or crop.size == 0:
        return False
    g = crop.astype(np.float32).mean(2) if crop.ndim == 3 else crop.astype(np.float32)
    paper, dark = float(np.median(g)), float(np.percentile(g, 1))
    if paper - dark < 40:  # no clear ink
        return False
    ink = g < (paper + dark) / 2
    H, W = ink.shape
    rows = np.flatnonzero(ink.sum(1) > 0.02 * W)
    if len(rows) == 0:
        return False
    band = rows[-1] - rows[0] + 1
    cols = ink[rows[0]:rows[-1] + 1].any(0)
    span = np.flatnonzero(cols)
    return bool(band <= 0.2 * H and (span[-1] - span[0] + 1) >= 0.5 * W)


def keep_printed(printed: str, handwriting: str, crop: np.ndarray | None = None) -> tuple[bool, str]:
    """Must the printed reading stay, whatever the handwriting reader's confidence? (decision, reason)

    The handwriting reader (TrOCR) invents words for lines without letters: it read a row of dashes
    as "sentence Valuable" with confidence 0.63 (IAM test line 02658). So when the printed reader saw
    only dashes, dots or rules (2 or more of them), or saw no letters or digits on a crop whose ink
    is a thin horizontal band, an alphabetic handwriting reading does not replace it."""
    if not _LETTER.search(handwriting or ""):
        return False, ""
    compact = "".join((printed or "").split())
    if _RULE_TEXT.fullmatch(compact):
        return True, "printed reading is a rule / row of dashes"
    if not _ALNUM.search(compact) and crop is not None and looks_like_rule(crop):
        return True, "the crop holds a horizontal rule"
    return False, ""


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

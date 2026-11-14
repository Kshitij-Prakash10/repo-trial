"""Handwriting recogniser: TrOCR (line images -> text with a confidence per line).

The model is microsoft/trocr-small-handwritten at a pinned revision, or a fine-tuned copy in
checkpoints/ when one exists (see HW_DIR). It reads one text line at a time.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from ocr.common import ROOT

HW_NAME = "microsoft/trocr-small-handwritten"
HW_REVISION = None  # set by _resolve() from checkpoints/handwriting_v1/source.json when present
HW_DIR = os.path.join(ROOT, "checkpoints", "handwriting_v1")
MAX_TOKENS = 96
NO_REPEAT = 4  # stops the decoder from looping on short or unclear lines


LINE_H = 96  # px; cleaned lines are stored and fed at up to this height


def prep_line(crop: np.ndarray) -> np.ndarray:
    """Make a line crop look like a scanned line: grey, paper flattened to white, ink darkened.

    Photographed handwriting (coloured ink, shadows) is far from what the model was pretrained on;
    this step alone cut its error on GNHK validation lines by about a third. Returns RGB uint8."""
    import cv2
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    if g.shape[0] > LINE_H:
        g = cv2.resize(g, (max(8, int(round(g.shape[1] * LINE_H / g.shape[0]))), LINE_H), interpolation=cv2.INTER_AREA)
    k = max(15, (g.shape[0] // 2) | 1)
    bg = cv2.morphologyEx(cv2.medianBlur(g, 5), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(np.float32)
    n = np.clip(g.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255)
    lo = float(np.percentile(n, 2))
    n = np.clip((n - lo) / max(255 - lo, 1) * 255, 0, 255).astype(np.uint8)
    return np.repeat(n[..., None], 3, 2)


class HandwritingReader:
    """Wraps processor + model. `read` takes RGB arrays or PIL images of single text lines."""

    def __init__(self, path_or_name: str | None = None, revision: str | None = None, local_only: bool = False):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        if path_or_name is None:
            path_or_name = HW_DIR if os.path.exists(os.path.join(HW_DIR, "config.json")) else HW_NAME
        kw = {"local_files_only": local_only}
        if revision and not os.path.isdir(path_or_name):
            kw["revision"] = revision
        self.processor = TrOCRProcessor.from_pretrained(path_or_name, use_fast=False, **kw)
        self.model = VisionEncoderDecoderModel.from_pretrained(path_or_name, **kw).eval()
        self.name = path_or_name

    @torch.no_grad()
    def read(self, lines: list, batch_size: int = 8, num_beams: int = 1, prep: bool = True) -> list[tuple[str, float]]:
        """(text, confidence) per line. Confidence is the geometric mean of the token probabilities.

        prep applies prep_line first (the model is fine-tuned on lines prepared that way)."""
        out = []
        arrs = [x if isinstance(x, np.ndarray) else np.asarray(x.convert("RGB")) for x in lines]
        ims = [Image.fromarray(prep_line(a) if prep else a) for a in arrs]
        for i in range(0, len(ims), batch_size):
            px = self.processor(images=ims[i:i + batch_size], return_tensors="pt").pixel_values
            gen = self.model.generate(px, max_new_tokens=MAX_TOKENS, num_beams=num_beams, output_scores=True,
                                      no_repeat_ngram_size=NO_REPEAT,
                                      return_dict_in_generate=True)
            texts = self.processor.batch_decode(gen.sequences, skip_special_tokens=True)
            scores = self.model.compute_transition_scores(gen.sequences, gen.scores,
                                                          getattr(gen, "beam_indices", None), normalize_logits=True)
            pad = self.processor.tokenizer.pad_token_id
            mask = gen.sequences[:, -scores.shape[1]:] != pad
            for t, s, m in zip(texts, scores, mask):
                s = s[m & torch.isfinite(s)]
                out.append((" ".join(t.split()), float(torch.exp(s.mean())) if len(s) else 0.0))
        return out

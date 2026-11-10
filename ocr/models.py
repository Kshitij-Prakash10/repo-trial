"""Build the text detector and recogniser (docTR, PyTorch) from pretrained or fine-tuned weights."""
from __future__ import annotations

import os

import torch

from ocr.common import ROOT

MODEL_DIR = os.path.join(ROOT, "models", "ocr")
DET_ARCH = "fast_base"  # chosen on validation, see NOTES.md
REC_ARCH = "crnn_mobilenet_v3_large"
DET_CKPT = os.path.join(MODEL_DIR, "det_fast_base_ft.pt")
REC_CKPT = os.path.join(MODEL_DIR, "rec_crnn_mobilenet_v3_large_ft.pt")


def load_det(arch: str = DET_ARCH, ckpt: str | None = None):
    """Detection model: docTR pretrained weights, then the fine-tuned checkpoint when one is given."""
    from doctr.models import detection
    model = getattr(detection, arch)(pretrained=True)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return model


def load_rec(arch: str = REC_ARCH, ckpt: str | None = None):
    """Recognition model: docTR pretrained weights, then the fine-tuned checkpoint when one is given."""
    from doctr.models import recognition
    model = getattr(recognition, arch)(pretrained=True)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return model


def det_predictor(model, batch_size: int = 2):
    """Page images (uint8 RGB arrays) -> relative word boxes; aspect ratio kept, as in training."""
    from doctr.models import detection_predictor
    return detection_predictor(arch=model.eval(), pretrained=False, batch_size=batch_size,
                               preserve_aspect_ratio=True, symmetric_pad=True)


def rec_predictor(model, batch_size: int = 128):
    """Word crops (uint8 RGB arrays) -> (text, confidence)."""
    from doctr.models import recognition_predictor
    return recognition_predictor(arch=model.eval(), pretrained=False, batch_size=batch_size)

"""Build the text detector and recogniser (docTR, PyTorch) from pretrained or fine-tuned weights."""
from __future__ import annotations

import os

import torch

from ocr.common import DET_CKPT, MODEL_DIR, REC_CKPT  # noqa: F401  (models.DET_CKPT etc. are used elsewhere)

DET_ARCH = "fast_base"  # chosen on validation, see NOTES.md
REC_ARCH = "crnn_mobilenet_v3_large"


def _load_checkpoint(model, ckpt: str):
    """Load a fine-tuned checkpoint ({"model": state_dict}); every weight must match (strict)."""
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=True)
    return model


def load_det(arch: str = DET_ARCH, ckpt: str | None = None):
    """Detection model. With a checkpoint file: built without any download and filled from the file
    (works offline). Without one: the docTR pretrained weights (downloaded on first use)."""
    from doctr.models import detection
    if ckpt and os.path.exists(ckpt):
        return _load_checkpoint(getattr(detection, arch)(pretrained=False, pretrained_backbone=False), ckpt)
    model = getattr(detection, arch)(pretrained=True)
    return _load_checkpoint(model, ckpt) if ckpt else model  # a named but missing file raises here


def load_rec(arch: str = REC_ARCH, ckpt: str | None = None):
    """Recognition model. With a checkpoint file: built without any download and filled from the file
    (works offline). Without one: the docTR pretrained weights (downloaded on first use)."""
    from doctr.models import recognition
    if ckpt and os.path.exists(ckpt):
        return _load_checkpoint(getattr(recognition, arch)(pretrained=False, pretrained_backbone=False), ckpt)
    model = getattr(recognition, arch)(pretrained=True)
    return _load_checkpoint(model, ckpt) if ckpt else model  # a named but missing file raises here


def det_predictor(model, batch_size: int = 2):
    """Page images (uint8 RGB arrays) -> relative word boxes; aspect ratio kept, as in training."""
    from doctr.models import detection_predictor
    return detection_predictor(arch=model.eval(), pretrained=False, batch_size=batch_size,
                               preserve_aspect_ratio=True, symmetric_pad=True)


def rec_predictor(model, batch_size: int = 128):
    """Word crops (uint8 RGB arrays) -> (text, confidence)."""
    from doctr.models import recognition_predictor
    return recognition_predictor(arch=model.eval(), pretrained=False, batch_size=batch_size)

"""HTTP API of the OCR pipeline (text from a scanned page). server.py registers `ocr_bp`.

POST /api/ocr         multipart: image (JPEG, PNG or WebP, at most 10 MB), optional scanner ("1" to run
                      the page-scanner stage first; off by default because the app sends a page that is
                      already scanned and flattened)
                      -> {"text", "lines": [{"text", "box", "confidence"}],
                          "words": [{"text", "confidence", "box", "line"}],
                          "page": {"width", "height"}, "seconds", "models"}
                      Boxes are [x0, y0, x1, y1] in pixels of the page; confidences are 0..1.
GET  /api/ocr/status  {"available": bool, "reason"?: str, "models"?: {"detector", "recogniser"}}

GET  /api/ocr/status  also returns "loaded" (whether the models are in memory) when available.

Errors are JSON ({"error": "..."}): 400 bad input, 413 too large, 503 when OCR cannot run here or is
busy (then with a Retry-After header).

The heavy packages (torch, python-doctr; requirements-ocr.txt) are optional and are imported only on
the first POST, never when the server starts. The models are loaded once and one page is read at a
time (a lock), because this runs on one CPU. At most MAX_WAITING requests are held at once and none
waits longer than LOCK_WAIT seconds; beyond that the endpoint answers 503 "busy". Whether OCR can run is
checked after the file header is read and before the image is decoded. Nothing is downloaded: when the docTR weights are not in the local
cache, the endpoint answers 503 instead of fetching them.
"""
from __future__ import annotations

import glob
import importlib.util
import io
import os
import threading
import time
import warnings
from typing import Any, Optional, Tuple

from flask import Blueprint, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError

ocr_bp = Blueprint("ocr", __name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAX_BYTES = 10 * 1024 * 1024  # upload limit, bytes (same as /api/scan)
MAX_PIXELS = 50_000_000  # decoded size limit for JPEG, pixels (48 MP phone photos)
MAX_PIXELS_OTHER = 25_000_000  # the same for PNG and WebP (a small file can declare a huge image)
MAX_SIDE = 12_000  # px, the longest side accepted
MAX_ASPECT = 20  # long side / short side; beyond this it is not a page
MAX_WAITING = 3  # requests held at once (one reading, the others waiting); more are told to retry
LOCK_WAIT = 60.0  # seconds a request waits for its turn before it is told to retry
RETRY_AFTER = 10  # seconds, sent with a "busy" answer
LOAD_RETRY = 60.0  # seconds after a failed model load before the next request may try again
FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}  # MPO: JPEG variant some phones write
PACKAGES = ("torch", "doctr")  # import names of the optional dependencies
SETUP_HINT = "pip install -r requirements-ocr.txt"

# The same files ocr/models.py uses; named here so that the status check does not import torch.
MODEL_DIR = os.path.join(ROOT, "models", "ocr")
DET_CKPT = os.path.join(MODEL_DIR, "det_fast_base_ft.pt")
REC_CKPT = os.path.join(MODEL_DIR, "rec_crnn_mobilenet_v3_large_ft.pt")
PRETRAINED = {"detector": "fast_base-*.pt", "recogniser": "crnn_mobilenet_v3_large_pt-*.pt"}  # docTR cache

_lock = threading.Lock()  # guards loading the models and running them
_pipeline: Any = None  # ocr.pipeline.OcrPipeline once loaded
_load_error: Optional[str] = None  # why the last load failed, shown by the status endpoint
_load_error_at = 0.0  # time.monotonic() of that failure; it stops counting after LOAD_RETRY seconds
_slots = threading.BoundedSemaphore(MAX_WAITING)  # requests inside the endpoint


class OcrUnavailable(RuntimeError):
    """OCR cannot run on this server; the message is safe to show to the user."""


class OcrBusy(RuntimeError):
    """Other pages are being read; the caller should try again shortly."""


def _recent_load_error() -> Optional[str]:
    """The last load failure while it is recent. After LOAD_RETRY seconds a new attempt is allowed, so a
    passing problem (for example too little free memory) does not switch reading text off until restart."""
    if _load_error and time.monotonic() - _load_error_at < LOAD_RETRY:
        return _load_error
    return None


def _err(msg: str, code: int):
    return jsonify({"error": msg}), code


def missing_packages() -> list[str]:
    """Names of the optional packages that are not installed (nothing is imported)."""
    out = []
    for name in PACKAGES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(name)
    return out


def missing_weights() -> list[str]:
    """Stages ("detector", "recogniser") whose docTR pretrained weights are not in the local cache."""
    cache = os.environ.get("DOCTR_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "doctr")
    return [stage for stage, pattern in PRETRAINED.items()
            if not glob.glob(os.path.join(cache, "models", pattern))]


def model_names() -> dict:
    """Which weights each stage would use: "finetuned" when its checkpoint is in models/ocr."""
    return {"detector": "finetuned" if os.path.exists(DET_CKPT) else "pretrained",
            "recogniser": "finetuned" if os.path.exists(REC_CKPT) else "pretrained"}


def availability() -> Tuple[bool, Optional[str]]:
    """(available, reason). The reason is one short sentence for the user when OCR cannot run."""
    missing = missing_packages()
    if missing:
        return False, f"Reading text needs extra packages on the server ({SETUP_HINT})."
    stages = missing_weights()
    if stages:
        return False, ("The text-reading model files are not on this computer yet "
                       f"({' and '.join(stages)}). See \"OCR scanner\" in README.md.")
    recent = _recent_load_error()
    if recent:
        return False, recent
    return True, None


def get_pipeline() -> Any:
    """The loaded pipeline (loads it on first use). Call with `_lock` held. Raises OcrUnavailable."""
    global _pipeline, _load_error, _load_error_at
    if _pipeline is not None:
        return _pipeline
    ok, reason = availability()
    if not ok:
        raise OcrUnavailable(reason)
    try:
        from ocr.pipeline import OcrPipeline  # imports torch and docTR: slow, so only here
        names = model_names()
        weights = "finetuned" if "finetuned" in names.values() else "pretrained"
        _pipeline = OcrPipeline.load(weights, use_scanner=True)
        _load_error = None
    except Exception as err:  # broken install, unreadable checkpoint, ...
        _load_error = f"The text-reading models could not be loaded ({type(err).__name__})."
        _load_error_at = time.monotonic()
        raise OcrUnavailable(_load_error) from err
    return _pipeline


def read_page(image: Image.Image, use_scanner: bool = False) -> dict:
    """Run the OCR pipeline on one page. One call at a time; raises OcrUnavailable or OcrBusy."""
    if not _lock.acquire(timeout=LOCK_WAIT):
        raise OcrBusy("Other pages are still being read. Try again in a moment.")
    try:
        pipe = get_pipeline()
        pipe.use_scanner = bool(use_scanner)  # per request; safe because the lock is held
        return pipe.run(image)
    finally:
        _lock.release()


def image_size_problem(fmt: str, w: int, h: int) -> Optional[str]:
    """None when a w x h image of this format is small enough to decode safely, else a message."""
    limit = MAX_PIXELS if fmt in ("JPEG", "MPO") else MAX_PIXELS_OTHER
    if w < 1 or h < 1 or w * h > limit or max(w, h) > MAX_SIDE:
        return "image has too many pixels"
    if max(w, h) > MAX_ASPECT * min(w, h):
        return "image is too narrow to be a page"
    return None


def _busy(msg: str):
    resp = jsonify({"error": msg})
    resp.status_code = 503
    resp.headers["Retry-After"] = str(RETRY_AFTER)
    return resp


def _file_size(f) -> int:
    stream = f.stream
    try:
        pos = stream.tell()
        stream.seek(0, os.SEEK_END)
        n = stream.tell()
        stream.seek(pos)
        return n
    except (AttributeError, OSError):
        return len(f.read())


def _flag(value: Optional[str]) -> Optional[bool]:
    """Form flag: "" / missing -> False, 1/true/yes/on -> True, 0/false/no/off -> False, else None."""
    v = (value or "").strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return None


@ocr_bp.before_app_request
def _allow_upload_size():
    """Let /api/ocr bodies through up to the 10 MB file limit plus form overhead."""
    if request.path == "/api/ocr":
        request.max_content_length = MAX_BYTES + 512 * 1024


@ocr_bp.errorhandler(413)
def _too_large(err):
    return _err("file is larger than 10 MB", 413)


@ocr_bp.post("/api/ocr")
def ocr_route():
    """Read the text on one uploaded page image."""
    if not _slots.acquire(blocking=False):
        return _busy("Other pages are still being read. Try again in a moment.")
    try:
        return _read_upload()
    except OcrBusy as err:
        return _busy(str(err))
    finally:
        _slots.release()


def _read_upload():
    f = request.files.get("image")
    if f is None:
        return _err("missing file field 'image'", 400)
    use_scanner = _flag(request.form.get("scanner"))
    if use_scanner is None:
        return _err("scanner must be 1 or 0", 400)
    size = _file_size(f)
    if size > MAX_BYTES:
        return _err("file is larger than 10 MB", 413)
    if size == 0:
        return _err("file 'image' is empty", 400)
    data = f.stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return _err("file is larger than 10 MB", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data))
            if im.format not in FORMATS:
                return _err("file must be a JPEG, PNG or WebP image", 400)
            problem = image_size_problem(im.format, *im.size)
            if problem:
                return _err(problem, 400)
            # Only the file header has been read so far. Decoding can take a lot of memory and time, so
            # it is not done when OCR cannot run here anyway.
            if _pipeline is None:
                ok, reason = availability()
                if not ok:
                    return _err(reason, 503)
            im = ImageOps.exif_transpose(im)
            im.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return _err("file is not a readable image", 400)
    except Exception:  # truncated or malformed files, broken EXIF
        return _err("file is not a readable image", 400)
    try:
        res = read_page(im, use_scanner)
    except OcrUnavailable as err:
        return _err(str(err), 503)
    return jsonify({
        "text": res["text"],
        "lines": [{"text": ln["text"], "box": ln["box"], "confidence": ln["confidence"]} for ln in res["lines"]],
        "words": [{"text": w["text"], "confidence": w["confidence"], "box": w["box"], "line": w.get("line")}
                  for w in res["words"]],
        "page": {"width": res["page"]["width"], "height": res["page"]["height"]},
        "seconds": res["seconds"],
        "models": res.get("models"),
    })


@ocr_bp.get("/api/ocr/status")
def ocr_status():
    """Whether text can be read on this server, and if not, why (no heavy import)."""
    ok, reason = availability()
    if not ok:
        return jsonify({"available": False, "reason": reason})
    names = _pipeline.name if _pipeline is not None and isinstance(_pipeline.name, dict) else model_names()
    return jsonify({"available": True, "models": names, "loaded": _pipeline is not None})

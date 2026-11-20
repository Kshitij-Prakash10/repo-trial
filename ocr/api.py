"""HTTP API of the OCR pipeline (text from a scanned page). server.py registers `ocr_bp`.

POST /api/ocr         multipart: image (JPEG, PNG or WebP, at most 10 MB) and two optional flags (1 / 0):
                      scanner  run the page-scanner stage first (off by default because the app sends
                               a page that is already scanned and flattened)
                      deskew   straighten slightly tilted text before reading (off by default; only the
                               "v2" pipeline can do it, with "v1" the flag is accepted and ignored)
                      -> {"text",
                          "lines": [{"text", "box", "confidence", "low_confidence", "confirm_numbers", "script"}],
                          "words": [{"text", "confidence", "box", "line"}],
                          "page": {"width", "height"}, "seconds", "models", "pipeline": "v2" | "v1",
                          "preprocessing": {"quarter_turns", "orientation_confidence", "deskew_degrees",
                                            "flip_check", "deskew_requested", "deskew_applied"}}
                      Boxes are [x0, y0, x1, y1] in pixels of the page; confidences are 0..1.
                      confirm_numbers: a reading of the line holds a number or a dose unit,
                      so a person must check it against the page whatever its confidence.
GET  /api/ocr/status  {"available": bool, "reason"?: str, "models"?: {"detector", "recogniser"},
                       "loaded"?: bool, "pipeline": "v2" | "v1",
                       "capabilities": {"handwriting", "orientation", "deskew"}, "pipeline_reason"?: str}
                      "loaded" (whether the models are in memory) and "models" come only when available.
                      "pipeline" is what a read would use (or uses, once loaded); "pipeline_reason" says
                      why it is not "v2".

Two pipelines. "v2" (ocr.scan.Scanner) turns the page upright, reads handwriting and flags handwritten
numbers; it needs the `transformers` package and the three checkpoints in checkpoints/. "v1"
(ocr.pipeline.OcrPipeline) reads printed text only and is used when v2 cannot be, including when v2
fails to load although everything seemed to be there.

Errors are JSON ({"error": "..."}): 400 bad input, 413 too large, 503 when OCR cannot run here or is
busy (then with a Retry-After header).

The heavy packages (torch, python-doctr: requirements-ocr.txt; transformers:
requirements-handwriting.txt) are optional and are imported only on the first POST, never when the
server starts. The models are loaded once and one page is read at a time (a lock), because this runs on
one CPU; with v2 they take about 1.3 GB of memory. At most MAX_WAITING requests are held at once and none
waits longer than LOCK_WAIT seconds; beyond that the endpoint answers 503 "busy". Whether OCR can run is
checked after the file header is read and before the image is decoded. Nothing is downloaded: when a
model file that is needed is not on this computer, the endpoint answers 503 instead of fetching it.
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
MIN_SIDE = 16  # px, the shortest side accepted (the same rule as /api/scan)
TOO_SMALL = f"image is too small: each side must be at least {MIN_SIDE} pixels"
LOW_CONFIDENCE = 0.5  # lines below this are flagged (the same value as ocr.scan.LOW_CONFIDENCE)
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

# What the v2 pipeline (ocr.scan.Scanner) needs on top of v1: the same files that ocr/orientation.py,
# ocr/scriptclf.py and ocr/handwriting.py load.
V2_PACKAGES = ("transformers",)
V2_SETUP_HINT = "pip install -r requirements-handwriting.txt"
CKPT_DIR = os.path.join(ROOT, "checkpoints")
V2_FILES = {"orientation": os.path.join(CKPT_DIR, "orientation_v1", "orientation.pt"),
            "printed / handwritten routing": os.path.join(CKPT_DIR, "scriptclf_v1", "scriptclf.pt"),
            "handwriting": os.path.join(CKPT_DIR, "handwriting_v1", "config.json")}

_lock = threading.Lock()  # guards loading the models and running them
_pipeline: Any = None  # ocr.scan.Scanner (v2) or ocr.pipeline.OcrPipeline (v1) once loaded
_v2_error: Optional[str] = None  # why v2 failed to load in this process (v1 was loaded instead)
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


def _not_installed(names: Tuple[str, ...]) -> list[str]:
    """Those of the import names that are not installed (nothing is imported)."""
    out = []
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(name)
    return out


def missing_packages() -> list[str]:
    """Names of the optional packages that are not installed (nothing is imported)."""
    return _not_installed(PACKAGES)


def missing_weights() -> list[str]:
    """Stages ("detector", "recogniser") that have no weights on this computer.

    A stage with a fine-tuned checkpoint in models/ocr is complete on its own. Only a stage without
    one needs docTR's pretrained file from the local cache."""
    cache = os.environ.get("DOCTR_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "doctr")
    ckpt = {"detector": DET_CKPT, "recogniser": REC_CKPT}  # looked up here so tests can replace the paths
    return [stage for stage, pattern in PRETRAINED.items()
            if not os.path.exists(ckpt[stage]) and not glob.glob(os.path.join(cache, "models", pattern))]


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


def v2_problem() -> Optional[str]:
    """None when the v2 pipeline can be tried, else one short sentence why not. Imports nothing."""
    missing = _not_installed(V2_PACKAGES)
    if missing:
        return (f"Handwriting and turned pages need the {' and '.join(missing)} package on the server "
                f"({V2_SETUP_HINT}).")
    files = [stage for stage, path in V2_FILES.items() if not os.path.exists(path)]
    if files:
        return f"The model files for {', '.join(files)} are not in the checkpoints folder."
    return _v2_error


def is_v2(pipe: Any) -> bool:
    """Whether a loaded pipeline is the v2 Scanner (it wraps the printed pipeline as `.ocr`)."""
    return hasattr(pipe, "ocr") and hasattr(pipe, "do_deskew")


def pipeline_info() -> dict:
    """{"pipeline", "capabilities", "pipeline_reason"?}: what is loaded, or what a read would load."""
    if _pipeline is not None:
        v2 = is_v2(_pipeline)
        reason = None if v2 else (v2_problem() or "The printed-text reader is loaded.")
    else:
        reason = v2_problem()
        v2 = reason is None
    if v2 and _pipeline is not None:  # a Scanner can be built with stages switched off
        caps = {"handwriting": getattr(_pipeline, "hw", None) is not None,
                "orientation": getattr(_pipeline, "orient", None) is not None, "deskew": True}
    else:
        caps = {"handwriting": v2, "orientation": v2, "deskew": v2}
    out = {"pipeline": "v2" if v2 else "v1", "capabilities": caps}
    if reason:
        out["pipeline_reason"] = reason
    return out


def _load_v2() -> Any:
    from ocr.scan import Scanner  # imports torch, docTR and transformers: slow, so only here
    return Scanner(deskew=False, page_scanner=True)


def _load_v1() -> Any:
    from ocr.pipeline import OcrPipeline  # imports torch and docTR: slow, so only here
    names = model_names()
    weights = "finetuned" if "finetuned" in names.values() else "pretrained"
    return OcrPipeline.load(weights, use_scanner=True)


def get_pipeline() -> Any:
    """The loaded pipeline (loads it on first use). Call with `_lock` held. Raises OcrUnavailable.

    v2 is tried when its packages and files are there. If it then fails to load, the reason is kept
    for the status endpoint and v1 is loaded instead, so the request is still answered (printed text
    only). v2 is not tried again until the server is restarted."""
    global _pipeline, _load_error, _load_error_at, _v2_error
    if _pipeline is not None:
        return _pipeline
    ok, reason = availability()
    if not ok:
        raise OcrUnavailable(reason)
    if v2_problem() is None:
        try:
            _pipeline = _load_v2()
            _load_error = None
            return _pipeline
        except Exception as err:  # broken install, unreadable checkpoint, not enough memory, ...
            _v2_error = ("The handwriting and orientation models could not be loaded "
                         f"({type(err).__name__}); printed text only until the server is restarted.")
    try:
        _pipeline = _load_v1()
        _load_error = None
    except Exception as err:  # broken install, unreadable checkpoint, ...
        _load_error = f"The text-reading models could not be loaded ({type(err).__name__})."
        _load_error_at = time.monotonic()
        raise OcrUnavailable(_load_error) from err
    return _pipeline


def read_page(image: Image.Image, use_scanner: bool = False, deskew: bool = False) -> dict:
    """Run the OCR pipeline on one page. One call at a time; raises OcrUnavailable or OcrBusy.

    Returns the pipeline's result plus "pipeline" ("v2" or "v1") and, under "preprocessing",
    "deskew_requested" and "deskew_applied". The per-request settings live on the shared model object,
    so they are set while the lock is held and put back to off before it is released."""
    if not _lock.acquire(timeout=LOCK_WAIT):
        raise OcrBusy("Other pages are still being read. Try again in a moment.")
    try:
        pipe = get_pipeline()
        v2 = is_v2(pipe)
        printed = pipe.ocr if v2 else pipe  # the object that holds use_scanner
        printed.use_scanner = bool(use_scanner)
        if v2:
            pipe.do_deskew = bool(deskew)
        try:
            from scanner.api import to_rgb8  # transparent areas -> white, 16-bit grey -> 8 bits
            res = dict(pipe.run(to_rgb8(image)))
        finally:
            printed.use_scanner = False
            if v2:
                pipe.do_deskew = False
        pre = dict(res.get("preprocessing") or {})
        pre["deskew_requested"] = bool(deskew)
        pre["deskew_applied"] = bool(v2 and deskew)
        res["preprocessing"] = pre
        res["pipeline"] = "v2" if v2 else "v1"
        return res
    finally:
        _lock.release()


def image_size_problem(fmt: str, w: int, h: int) -> Optional[str]:
    """None when a w x h image of this format is small enough to decode safely, else a message."""
    limit = MAX_PIXELS if fmt in ("JPEG", "MPO") else MAX_PIXELS_OTHER
    if min(w, h) < MIN_SIDE:
        return TOO_SMALL
    if w * h > limit or max(w, h) > MAX_SIDE:
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
    deskew = _flag(request.form.get("deskew"))
    if deskew is None:
        return _err("deskew must be 1 or 0", 400)
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
        res = read_page(im, use_scanner, deskew)
    except OcrUnavailable as err:
        return _err(str(err), 503)
    return jsonify({
        "text": res["text"],
        "lines": [_line(ln) for ln in res["lines"]],
        "words": [{"text": w["text"], "confidence": w["confidence"], "box": w["box"], "line": w.get("line")}
                  for w in res["words"]],
        "page": {"width": res["page"]["width"], "height": res["page"]["height"]},
        "seconds": res["seconds"],
        "models": res.get("models"),
        "pipeline": res["pipeline"],
        "preprocessing": _preprocessing(res["preprocessing"]),
    })


def _line(ln: dict) -> dict:
    """Preserve alternative-reader warnings and check the displayed text, including v1 results."""
    from ocr.common import has_number_or_unit
    conf = ln["confidence"]
    low = ln.get("low_confidence")
    return {"text": ln["text"], "box": ln["box"], "confidence": conf,
            "low_confidence": bool(conf < LOW_CONFIDENCE) if low is None else bool(low),
            "confirm_numbers": bool(ln.get("confirm_numbers")) or has_number_or_unit(ln["text"]),
            "script": ln.get("script") or "printed"}


def _preprocessing(pre: dict) -> dict:
    """What was done to the page before reading (zeros and nulls when the v1 pipeline ran)."""
    flip = pre.get("flip_check")
    return {"quarter_turns": int(pre.get("quarter_turns") or 0),
            "orientation_confidence": pre.get("orientation_confidence"),
            "deskew_degrees": float(pre.get("deskew_degrees") or 0.0),
            "flip_check": flip if isinstance(flip, dict) else None,
            "deskew_requested": bool(pre.get("deskew_requested")),
            "deskew_applied": bool(pre.get("deskew_applied"))}


@ocr_bp.get("/api/ocr/status")
def ocr_status():
    """Whether text can be read on this server, and if not, why (no heavy import)."""
    ok, reason = availability()
    info = pipeline_info()
    if not ok:
        return jsonify({"available": False, "reason": reason, **info})
    printed = getattr(_pipeline, "ocr", _pipeline)  # the v2 Scanner wraps the printed pipeline
    name = getattr(printed, "name", None)
    names = name if isinstance(name, dict) else model_names()
    return jsonify({"available": True, "models": names, "loaded": _pipeline is not None, **info})

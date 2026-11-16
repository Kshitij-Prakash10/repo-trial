"""HTTP API of the OCR pipeline (text from a scanned page). server.py registers `ocr_bp`.

POST /api/ocr         multipart: image (JPEG, PNG or WebP, at most 10 MB), optional scanner ("1" to run
                      the page-scanner stage first; off by default because the app sends a page that is
                      already scanned and flattened)
                      -> {"text", "lines": [{"text", "box", "confidence", "script", "low_confidence"}],
                          "words": [{"text", "confidence", "box", "line"}],
                          "page": {"width", "height"}, "seconds", "models", "handwriting",
                          "preprocessing": {"quarter_turns", "orientation_confidence",
                                            "deskew_degrees", "flip_check"?} or null}
                      Boxes are [x0, y0, x1, y1] in pixels of the page that was read (after any turn);
                      confidences are 0..1. "script" is "printed" or "handwritten" (the reader whose
                      text was kept); "low_confidence" marks lines below 0.5. "words" are the printed
                      reader's words; "line" is an index into "lines", or null.
GET  /api/ocr/status  {"available": bool, "reason"?: str, "models"?: {"detector", "recogniser"},
                       "loaded"?: bool, "handwriting": bool, "handwriting_reason"?: str}

Errors are JSON ({"error": "..."}): 400 bad input, 413 too large, 503 when OCR cannot run here.

Two modes. With the handwriting packages (requirements-handwriting.txt) and the three checkpoints in
checkpoints/, pages are read by ocr.scan.Scanner: turned pages are put upright and handwritten lines
go to the handwriting reader ("handwriting": true). Otherwise the printed-only pipeline
(ocr.pipeline.OcrPipeline) reads the page as before ("handwriting": false, "preprocessing": null,
every line "printed"), and the status says why in "handwriting_reason".

The heavy packages (torch, python-doctr; requirements-ocr.txt) are optional and are imported only on
the first POST, never when the server starts. The models are loaded once and one page is read at a
time (a lock), because this runs on one CPU. Nothing is downloaded: when the docTR weights are not in
the local cache, the endpoint answers 503 instead of fetching them, and the handwriting mode is used
only when every one of its model files is already on disk.
"""
from __future__ import annotations

import glob
import importlib.util
import io
import os
import threading
import warnings
from typing import Any, Optional, Tuple

from flask import Blueprint, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError

ocr_bp = Blueprint("ocr", __name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAX_BYTES = 10 * 1024 * 1024  # upload limit, bytes (same as /api/scan)
MAX_PIXELS = 50_000_000  # decoded size limit, pixels (decompression-bomb guard)
FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}  # MPO: JPEG variant some phones write
PACKAGES = ("torch", "doctr")  # import names of the optional dependencies
SETUP_HINT = "pip install -r requirements-ocr.txt"

# The same files ocr/models.py uses; named here so that the status check does not import torch.
MODEL_DIR = os.path.join(ROOT, "models", "ocr")
DET_CKPT = os.path.join(MODEL_DIR, "det_fast_base_ft.pt")
REC_CKPT = os.path.join(MODEL_DIR, "rec_crnn_mobilenet_v3_large_ft.pt")
PRETRAINED = {"detector": "fast_base-*.pt", "recogniser": "crnn_mobilenet_v3_large_pt-*.pt"}  # docTR cache

# Handwriting and rotated pages (ocr/scan.py). The same files ocr/handwriting.py, ocr/orientation.py and
# ocr/scriptclf.py load; named here so that the status check does not import torch.
HW_PACKAGES = ("transformers", "sentencepiece", "safetensors")  # import names (requirements-handwriting.txt)
HW_SETUP_HINT = "pip install -r requirements-handwriting.txt"
CKPT_DIR = os.path.join(ROOT, "checkpoints")
HW_DIR = os.path.join(CKPT_DIR, "handwriting_v1")
HW_FILES = (os.path.join(HW_DIR, "config.json"), os.path.join(HW_DIR, "preprocessor_config.json"),
            os.path.join(HW_DIR, "tokenizer_config.json"), os.path.join(HW_DIR, "sentencepiece.bpe.model"),
            os.path.join(CKPT_DIR, "orientation_v1", "orientation.pt"),
            os.path.join(CKPT_DIR, "scriptclf_v1", "scriptclf.pt"))
HW_WEIGHTS = ("model.safetensors", "model-*.safetensors")  # one file, or shards, in HW_DIR
LOW_CONFIDENCE = 0.5  # lines below this are flagged; the same value as ocr.scan.LOW_CONFIDENCE

_lock = threading.Lock()  # guards loading the models and running them
_scanner: Any = None  # ocr.scan.Scanner once loaded (handwriting mode)
_pipeline: Any = None  # ocr.pipeline.OcrPipeline once loaded (printed-only mode)
_load_error: Optional[str] = None  # why the last load failed, shown by the status endpoint
_hw_load_error: Optional[str] = None  # why the handwriting mode could not be loaded (printed-only is used)


class OcrUnavailable(RuntimeError):
    """OCR cannot run on this server; the message is safe to show to the user."""


def _err(msg: str, code: int):
    return jsonify({"error": msg}), code


def missing_packages(names: Optional[Tuple[str, ...]] = None) -> list[str]:
    """Names of the optional packages (default: PACKAGES) that are not installed (nothing is imported)."""
    out = []
    for name in (PACKAGES if names is None else names):
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
    if _load_error:
        return False, _load_error
    return True, None


def missing_handwriting_files() -> list[str]:
    """Model files of the handwriting mode that are not on disk, as paths relative to the repository."""
    out = [os.path.relpath(path, ROOT) for path in HW_FILES if not os.path.exists(path)]
    if not any(glob.glob(os.path.join(HW_DIR, pattern)) for pattern in HW_WEIGHTS):
        out.append(os.path.relpath(os.path.join(HW_DIR, HW_WEIGHTS[0]), ROOT))
    return out


def handwriting_availability() -> Tuple[bool, Optional[str]]:
    """(usable, reason) for the handwriting and rotated-page mode; nothing is imported or loaded.

    The reason is one short sentence when the printed-only pipeline is (or would be) used instead."""
    if _scanner is not None:
        return True, None
    if missing_packages(HW_PACKAGES):
        return False, ("Handwriting and turned pages need extra packages on the server "
                       f"({HW_SETUP_HINT}); printed text is read without them.")
    files = missing_handwriting_files()
    if files:
        return False, ("The handwriting model files are not on this computer "
                       f"({', '.join(files)}); printed text is read without them.")
    if _hw_load_error:
        return False, _hw_load_error
    if _pipeline is not None:  # a printed-only pipeline is in use (it was loaded before the mode became usable)
        return False, "The printed-only reader is loaded; restart the server to read handwriting."
    return True, None


def _load_scanner(weights: str) -> Any:
    """Build ocr.scan.Scanner from local files only (imports torch, docTR and transformers: slow)."""
    # Every file is on disk (checked by the caller), so nothing is fetched; these two switches make the
    # Hugging Face libraries refuse the network outright in case a file turns out to be unusable.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from ocr.scan import Scanner
    return Scanner(weights=weights, page_scanner=True)


def get_pipeline() -> Any:
    """The loaded reader (loads it on first use). Call with `_lock` held. Raises OcrUnavailable.

    This is ocr.scan.Scanner when the handwriting mode is usable, else ocr.pipeline.OcrPipeline. When
    the Scanner cannot be loaded the reason is kept for the status and the printed pipeline is used."""
    global _scanner, _pipeline, _load_error, _hw_load_error
    if _scanner is not None:
        return _scanner
    if _pipeline is not None:
        return _pipeline
    if missing_packages() or missing_weights():
        raise OcrUnavailable(availability()[1])
    names = model_names()
    weights = "finetuned" if "finetuned" in names.values() else "pretrained"
    if handwriting_availability()[0]:
        try:
            _scanner = _load_scanner(weights)
            _load_error = None
            return _scanner
        except Exception as err:  # broken install, unreadable checkpoint, ...: read printed text only
            _hw_load_error = (f"The handwriting models could not be loaded ({type(err).__name__}); "
                              "printed text is read without them.")
    try:
        from ocr.pipeline import OcrPipeline  # imports torch and docTR: slow, so only here
        _pipeline = OcrPipeline.load(weights, use_scanner=True)
        _load_error = None
    except Exception as err:  # broken install, unreadable checkpoint, ...
        _load_error = f"The text-reading models could not be loaded ({type(err).__name__})."
        raise OcrUnavailable(_load_error) from err
    return _pipeline


def read_page(image: Image.Image, use_scanner: bool = False) -> dict:
    """Read one page with the loaded reader. One call at a time; raises OcrUnavailable.

    Returns the reader's result plus "handwriting": True when ocr.scan.Scanner read the page."""
    with _lock:
        reader = get_pipeline()
        with_hw = reader is _scanner
        # the page-scanner stage is set per request; safe because the lock is held
        (reader.ocr if with_hw else reader).use_scanner = bool(use_scanner)
        return {**reader.run(image), "handwriting": with_hw}


def public_result(res: dict) -> dict:
    """The JSON answer of POST /api/ocr for one result of `read_page` (either mode)."""
    lines, line_of = [], {}
    for ln in res["lines"]:
        for w in ln.get("words") or []:
            line_of[w] = len(lines)
        lines.append({"text": ln["text"], "box": ln["box"], "confidence": ln["confidence"],
                      "script": ln.get("script", "printed"),
                      "low_confidence": bool(ln.get("low_confidence", ln["confidence"] < LOW_CONFIDENCE))})
    # A word's "line" must point into the lines above: the Scanner drops lines without text.
    words = [{"text": w["text"], "confidence": w["confidence"], "box": w["box"],
              "line": line_of.get(i) if line_of else w.get("line")} for i, w in enumerate(res["words"])]
    models = res.get("models")
    if isinstance(models, dict) and models.get("handwriting"):
        # the reader's name is the folder it was loaded from; keep its name, not the path on this computer
        models = {**models, "handwriting": os.path.basename(os.path.normpath(str(models["handwriting"])))}
    return {
        "text": res["text"],
        "lines": lines,
        "words": words,
        "page": {"width": res["page"]["width"], "height": res["page"]["height"]},
        "seconds": res["seconds"],
        "models": models,
        "handwriting": bool(res.get("handwriting")),
        "preprocessing": res.get("preprocessing"),
    }


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
            if im.size[0] * im.size[1] > MAX_PIXELS:
                return _err("image has too many pixels", 400)
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
    return jsonify(public_result(res))


@ocr_bp.get("/api/ocr/status")
def ocr_status():
    """Whether text can be read on this server, and if not, why (no heavy import)."""
    ok, reason = availability()
    if not ok:
        return jsonify({"available": False, "reason": reason, "handwriting": False})
    loaded = _scanner.ocr if _scanner is not None else _pipeline
    names = loaded.name if loaded is not None and isinstance(loaded.name, dict) else model_names()
    body = {"available": True, "models": names, "loaded": loaded is not None}
    body["handwriting"], hw_reason = handwriting_availability()
    if hw_reason:
        body["handwriting_reason"] = hw_reason
    return jsonify(body)

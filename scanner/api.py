"""HTTP API of the document scanner (contract: docs/PLAN.md). Agent A registers `scan_bp` in server.py.

POST /api/scan        multipart: image (JPEG, PNG or WebP, at most 10 MB), mode (color | gray | bw)
GET  /api/scan/status {"available", "model_version", "metrics"}
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import warnings

from flask import Blueprint, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError

from scanner import infer

scan_bp = Blueprint("scan", __name__)

MAX_BYTES = 10 * 1024 * 1024  # upload limit, bytes
MAX_PIXELS = 50_000_000  # decoded size limit, pixels (decompression-bomb guard)
WORK_SIDE = 2400  # px, the long side the photo is reduced to before scanning
FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}  # MPO: JPEG variant some phones write
MODES = ("color", "gray", "bw")


def _err(msg: str, code: int):
    return jsonify({"error": msg}), code


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


@scan_bp.before_app_request
def _allow_upload_size():
    """Let /api/scan bodies through up to the 10 MB file limit plus form overhead (app default is lower)."""
    if request.path == "/api/scan":
        request.max_content_length = MAX_BYTES + 512 * 1024


@scan_bp.errorhandler(413)
def _too_large(err):
    return _err("file is larger than 10 MB", 413)


@scan_bp.post("/api/scan")
def scan_route():
    """Scan one uploaded photo and return the corrected page as base64 PNG."""
    t0 = time.perf_counter()
    f = request.files.get("image")
    if f is None:
        return _err("missing file field 'image'", 400)
    mode = (request.form.get("mode") or "gray").strip().lower()
    if mode not in MODES:
        return _err(f"mode must be one of {', '.join(MODES)}", 400)
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
            ow, oh = im.size
            if (im.getexif().get(0x0112) or 1) in (5, 6, 7, 8):  # EXIF orientation swaps width and height
                ow, oh = oh, ow
            if im.format in ("JPEG", "MPO"):
                im.draft("RGB", (WORK_SIDE, WORK_SIDE))  # fast JPEG downscale while decoding
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return _err("file is not a readable image", 400)
    except Exception:  # truncated or malformed files, broken EXIF
        return _err("file is not a readable image", 400)
    # corners are reported in the pixels of the uploaded image after EXIF orientation (ow x oh)
    if max(im.size) > WORK_SIDE:
        im.thumbnail((WORK_SIDE, WORK_SIDE), Image.BOX)
    sx, sy = ow / im.size[0], oh / im.size[1]
    res = infer.scan(im, mode)
    corners = [[round(x * sx, 1), round(y * sy, 1)] for x, y in res["corners"]]
    buf = io.BytesIO()
    res["image"].save(buf, "PNG", compress_level=6 if mode == "bw" else 1)  # level 1: fast for photos
    out = res["image"]
    return jsonify({
        "page_found": bool(res["page_found"]), "corners": corners,
        "corners_norm": [[round(x / ow, 4), round(y / oh, 4)] for x, y in corners],
        "confidence": float(res["confidence"]), "method": res["method"],
        "width": out.size[0], "height": out.size[1],
        "image_png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "model_version": res["model_version"],
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)})


@scan_bp.get("/api/scan/status")
def scan_status():
    """Whether the trained scanner model is available, its version, and its held-out metrics."""
    model = infer.load_model()
    metrics = None
    try:
        with open(infer.METRICS_PATH) as fh:
            metrics = json.load(fh)
    except (OSError, ValueError):
        pass
    return jsonify({"available": model is not None,
                    "model_version": model.get("model_version") if model else None, "metrics": metrics})

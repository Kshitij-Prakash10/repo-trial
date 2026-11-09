"""Tests for the optional `corners` field of POST /api/scan (manual page corners, no detection).

All images are synthetic: a light page drawn on a dark background with Pillow.
"""
from __future__ import annotations

import base64
import io
import json

import pytest
from flask import Flask
from PIL import Image, ImageDraw

from scanner import infer
from scanner.api import scan_bp

PAGE = [[120, 80], [520, 110], [500, 690], [90, 660]]  # TL, TR, BR, BL in px of a 640 x 800 photo


@pytest.fixture()
def client(monkeypatch):
    def no_detection(*a, **k):
        raise AssertionError("detection must be skipped when corners are given")

    monkeypatch.setattr(infer, "scan", no_detection)
    app = Flask(__name__)
    app.register_blueprint(scan_bp)
    return app.test_client()


def _photo(w: int = 640, h: int = 800, page=PAGE, fmt: str = "PNG", **save_kw) -> io.BytesIO:
    im = Image.new("RGB", (w, h), (52, 44, 38))
    ImageDraw.Draw(im).polygon([tuple(p) for p in page], fill=(245, 243, 236))
    buf = io.BytesIO()
    im.save(buf, fmt, **save_kw)
    buf.seek(0)
    return buf


def _post(client, corners, mode: str = "gray", photo: io.BytesIO | None = None):
    text = corners if isinstance(corners, str) else json.dumps(corners)
    return client.post("/api/scan", data={"image": (photo or _photo(), "p.png"), "mode": mode, "corners": text},
                       content_type="multipart/form-data")


@pytest.mark.parametrize("mode, pil_mode", [("color", "RGB"), ("gray", "L"), ("bw", "L")])
def test_valid_corners_warp_without_detection(client, mode, pil_mode):
    r = _post(client, PAGE, mode)
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["method"] == "manual" and j["page_found"] is True and j["confidence"] is None
    assert j["corners"] == [[float(x), float(y)] for x, y in PAGE]
    assert j["corners_norm"][2] == [round(500 / 640, 4), round(690 / 800, 4)]
    assert set(j) == {"page_found", "corners", "corners_norm", "confidence", "method", "width", "height",
                      "image_png_b64", "model_version", "elapsed_ms"}
    out = Image.open(io.BytesIO(base64.b64decode(j["image_png_b64"])))
    assert out.format == "PNG" and out.size == (j["width"], j["height"]) and out.mode == pil_mode
    assert out.size[1] > out.size[0]  # portrait page
    assert isinstance(j["model_version"], str)


def test_counter_clockwise_corners_are_returned_tl_tr_br_bl(client):
    ccw = [PAGE[3], PAGE[2], PAGE[1], PAGE[0]]  # BL, BR, TR, TL: still a convex outline
    j = _post(client, ccw).get_json()
    assert j["corners"] == [[float(x), float(y)] for x, y in PAGE]


def test_corners_slightly_outside_the_image_are_accepted(client):
    r = _post(client, [[-30, -40], [660, -10], [700, 860], [-60, 820]])  # within 10 % of 640 x 800
    assert r.status_code == 200, r.get_json()


def test_corners_are_in_original_pixels_for_large_photos(client):
    w, h = 3600, 2400  # larger than the 2400 px working size, so the server downscales
    page = [[400, 300], [3200, 260], [3300, 2100], [300, 2150]]
    j = _post(client, page, photo=_photo(w, h, page, "JPEG", quality=80)).get_json()
    assert j["corners"] == [[float(x), float(y)] for x, y in page]
    assert j["width"] > j["height"]  # landscape page


def test_corners_use_exif_rotated_pixels(client):
    # Stored landscape 800 x 640 with EXIF orientation 6: displayed (and corners given) as portrait 640 x 800.
    exif = Image.Exif()
    exif[0x0112] = 6
    stored = Image.open(_photo()).transpose(Image.Transpose.ROTATE_90)
    buf = io.BytesIO()
    stored.save(buf, "JPEG", quality=90, exif=exif)
    buf.seek(0)
    r = _post(client, PAGE, photo=buf)
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["corners_norm"][1] == [round(520 / 640, 4), round(110 / 800, 4)]


@pytest.mark.parametrize("corners", [
    PAGE[:3],                                         # three points
    PAGE + [[10, 10]],                                # five points
    {"tl": [0, 0]},                                   # not an array
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]],      # not pairs
    [["120", 80], [520, 110], [500, 690], [90, 660]],  # strings
    [[True, 80], [520, 110], [500, 690], [90, 660]],  # booleans
    "[[NaN, 80], [520, 110], [500, 690], [90, 660]]",  # not finite
    "[[Infinity, 80], [520, 110], [500, 690], [90, 660]]",
])
def test_bad_shape_is_rejected(client, corners):
    r = _post(client, corners)
    assert r.status_code == 400
    assert "corner" in r.get_json()["error"]


@pytest.mark.parametrize("corners", [
    [[120, 80], [500, 690], [520, 110], [90, 660]],   # crossed (bow tie)
    [[120, 80], [520, 110], [250, 250], [90, 660]],   # one corner pushed inwards (concave)
    [[100, 100], [300, 100], [500, 100], [100, 500]],  # three points on one line
])
def test_non_convex_quad_is_rejected(client, corners):
    r = _post(client, corners)
    assert r.status_code == 400
    assert "convex" in r.get_json()["error"]


@pytest.mark.parametrize("corners", [
    [[120, 80], [520, 110], [500, 690], [-70, 660]],  # x more than 10 % left of the image
    [[120, 80], [720, 110], [500, 690], [90, 660]],   # x more than 10 % right
    [[120, -90], [520, 110], [500, 690], [90, 660]],  # y more than 10 % above
    [[120, 80], [520, 110], [500, 900], [90, 660]],   # y more than 10 % below
])
def test_out_of_bounds_is_rejected(client, corners):
    r = _post(client, corners)
    assert r.status_code == 400
    assert "within the image" in r.get_json()["error"]


@pytest.mark.parametrize("text", ["not json", "[[1, 2], [3, 4]", "{", "[[1,2],[3,4],[5,6],[7,8]] trailing"])
def test_bad_json_is_rejected(client, text):
    r = _post(client, text)
    assert r.status_code == 400
    assert "JSON" in r.get_json()["error"]


def test_empty_corners_field_means_detect(monkeypatch):
    calls = []

    def fake_scan(img, mode):
        calls.append(mode)
        w, h = img.size
        return {"page_found": False, "corners": [[0, 0], [w, 0], [w, h], [0, h]], "confidence": 0.1,
                "image": img.convert("L"), "model_version": "test", "method": "fallback"}

    monkeypatch.setattr(infer, "scan", fake_scan)
    app = Flask(__name__)
    app.register_blueprint(scan_bp)
    r = app.test_client().post("/api/scan", data={"image": (_photo(), "p.png"), "corners": ""},
                               content_type="multipart/form-data")
    assert r.status_code == 200 and calls == ["gray"]
    assert r.get_json()["method"] == "fallback"

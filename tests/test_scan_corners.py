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


@pytest.mark.parametrize("text", [
    "[[1" + "0" * 320 + ",0],[10,0],[10,10],[0,10]]",  # an integer too large for a float (under the length cap)
    "[" * 5000,  # nested too deeply for the JSON parser
    "[[0." + "0" * 500 + ", 0], [10, 0], [10, 10], [0, 10]]",  # valid, but far longer than 4 points need
])
def test_hostile_corners_are_400_not_500(client, text):
    r = _post(client, text)
    assert r.status_code == 400 and "corner" in r.get_json()["error"]


def _flat_png(w: int, h: int) -> io.BytesIO:
    """A one-colour PNG: a few kB on disk however many pixels it declares."""
    buf = io.BytesIO()
    Image.new("L", (w, h), 255).save(buf, "PNG", compress_level=9)
    buf.seek(0)
    return buf


@pytest.mark.parametrize("w, h, message", [
    (30000, 100, "too narrow"),  # a strip: few pixels, extreme shape (and over the side limit)
    (13000, 1000, "too many pixels"),  # one side over the limit
    (6000, 5000, "too many pixels"),  # 30 MP: allowed for JPEG, not for PNG (decoded at full size)
    (4000, 100, "too narrow"),  # 40:1
])
def test_oversized_or_extreme_images_are_rejected_before_decoding(monkeypatch, w, h, message):
    from scanner import api

    def no_decode(*a, **k):
        raise AssertionError("the image must be rejected from its header, before it is decoded")

    monkeypatch.setattr(api.ImageOps, "exif_transpose", no_decode)
    app = Flask(__name__)
    app.register_blueprint(scan_bp)
    r = app.test_client().post("/api/scan", data={"image": (_flat_png(w, h), "p.png")},
                               content_type="multipart/form-data")
    assert r.status_code == 400
    assert message in r.get_json()["error"] or "too many pixels" in r.get_json()["error"]


def test_image_size_rule():
    from scanner.api import image_size_problem
    assert image_size_problem("JPEG", 8000, 6000) is None  # a 48 MP phone photo
    assert image_size_problem("PNG", 4000, 3000) is None
    assert image_size_problem("PNG", 8000, 6000) is not None
    assert image_size_problem("JPEG", 1, 49_000_000) is not None
    assert image_size_problem("PNG", 2000, 99) is not None
    small = "image is too small: each side must be at least 16 pixels"
    for w, h in [(1, 1), (2, 2), (15, 15), (15, 100), (100, 15), (0, 0)]:
        assert image_size_problem("PNG", w, h) == small
        assert image_size_problem("JPEG", w, h) == small
    assert image_size_problem("PNG", 16, 16) is None
    assert image_size_problem("PNG", 16, 320) is None


# ---------------------------------------------------------------- the smallest image

@pytest.mark.parametrize("w, h", [(1, 1), (2, 2), (15, 40)])
@pytest.mark.parametrize("with_corners", [True, False])
def test_tiny_images_are_rejected(client, monkeypatch, w, h, with_corners):
    from scanner import api

    def no_decode(*a, **k):
        raise AssertionError("the image must be rejected from its header, before it is decoded")

    monkeypatch.setattr(api.ImageOps, "exif_transpose", no_decode)
    data = {"image": (_flat_png(w, h), "p.png"), "mode": "gray"}
    if with_corners:
        data["corners"] = json.dumps([[0, 0], [w, 0], [w, h], [0, h]])
    r = client.post("/api/scan", data=data, content_type="multipart/form-data")  # detection would raise (fixture)
    assert r.status_code == 400
    assert r.get_json()["error"] == "image is too small: each side must be at least 16 pixels"


def test_smallest_accepted_image_is_scanned(client):
    r = client.post("/api/scan", data={"image": (_flat_png(16, 16), "p.png"), "mode": "gray",
                                       "corners": json.dumps([[0, 0], [16, 0], [16, 16], [0, 16]])},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()


# ---------------------------------------------------------------- image modes (transparency, 16-bit grey)

W, H = 300, 400
FULL = [[0, 0], [W, 0], [W, H], [0, H]]  # the whole image as the page: no detection, no cropping
TEXT = (60, 100, 240, 130)  # x0, y0, x1, y1 of the dark "text" bar drawn on each page


def _scan_pixels(client, im: Image.Image, mode: str = "color", fmt: str = "PNG", **save_kw):
    """Upload `im` with the whole image as the page; (paper pixel, text pixel) of the returned page."""
    buf = io.BytesIO()
    im.save(buf, fmt, **save_kw)
    buf.seek(0)
    r = _post(client, FULL, mode, buf)
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    out = Image.open(io.BytesIO(base64.b64decode(j["image_png_b64"]))).convert("L")
    sx, sy = out.size[0] / W, out.size[1] / H
    paper = out.getpixel((int(20 * sx), int(300 * sy)))
    text = out.getpixel((int(150 * sx), int(115 * sy)))
    return paper, text


def _assert_dark_text_on_light_paper(paper: int, text: int):
    assert paper >= 200, f"paper came out dark ({paper})"
    assert text <= 60, f"text came out light ({text})"


@pytest.mark.parametrize("mode", ["color", "gray", "bw"])
def test_transparent_rgba_png_keeps_dark_text_on_white(client, mode):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))  # fully transparent "paper"
    ImageDraw.Draw(im).rectangle(TEXT, fill=(20, 20, 20, 255))
    _assert_dark_text_on_light_paper(*_scan_pixels(client, im, mode))


def test_16_bit_grey_png_keeps_its_contrast(client):
    import numpy as np
    a = np.full((H, W), 32768, np.uint16)  # a mid-grey page (half of 65535)
    a[TEXT[1]:TEXT[3], TEXT[0]:TEXT[2]] = 3000  # dark text
    im = Image.fromarray(a)
    assert im.mode.startswith("I;16")
    _assert_dark_text_on_light_paper(*_scan_pixels(client, im))


def test_grey_with_alpha_png(client):
    im = Image.new("LA", (W, H), (0, 0))
    ImageDraw.Draw(im).rectangle(TEXT, fill=(20, 255))
    _assert_dark_text_on_light_paper(*_scan_pixels(client, im))


def test_palette_png_with_a_transparent_colour(client):
    im = Image.new("P", (W, H), 0)
    im.putpalette([0, 0, 0, 20, 20, 20] + [0] * (254 * 3))  # 0: black (made transparent), 1: dark text
    ImageDraw.Draw(im).rectangle(TEXT, fill=1)
    _assert_dark_text_on_light_paper(*_scan_pixels(client, im, transparency=0))


def test_transparent_webp(client):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle(TEXT, fill=(20, 20, 20, 255))
    _assert_dark_text_on_light_paper(*_scan_pixels(client, im, fmt="WEBP", lossless=True))


def test_partly_transparent_surroundings_keep_their_colour(client):
    """The upload path makes the same choice as scanner.convert.to_rgb: a half see-through dark border
    around an opaque page stays dark (it is not lightened towards white), transparent pixels turn white."""
    import numpy as np
    from scanner.api import to_rgb8
    a = np.zeros((H, W, 4), np.uint8)
    a[..., :3], a[..., 3] = 20, 128
    a[50:350, 50:250] = (245, 245, 240, 255)
    a[0, 0] = (0, 0, 0, 0)
    out = to_rgb8(Image.fromarray(a, "RGBA"))
    assert out.getpixel((10, 10)) == (20, 20, 20) and out.getpixel((100, 100)) == (245, 245, 240)
    assert out.getpixel((0, 0)) == (255, 255, 255)


def test_to_rgb8_values():
    import numpy as np
    from scanner.api import to_rgb8
    rgb = Image.new("RGB", (4, 4), (10, 20, 30))
    assert to_rgb8(rgb) is rgb  # nothing to do
    assert to_rgb8(Image.new("RGBA", (4, 4), (0, 0, 0, 0))).getpixel((0, 0)) == (255, 255, 255)
    assert to_rgb8(Image.new("RGBA", (4, 4), (200, 0, 0, 255))).getpixel((0, 0)) == (200, 0, 0)
    assert to_rgb8(Image.new("RGBA", (4, 4), (0, 0, 0, 128))).getpixel((0, 0)) == (127, 127, 127)  # half see-through black
    assert to_rgb8(Image.new("L", (4, 4), 77)).getpixel((0, 0)) == (77, 77, 77)
    assert to_rgb8(Image.new("1", (4, 4), 1)).getpixel((0, 0)) == (255, 255, 255)
    for value, grey in [(0, 0), (65535, 255), (32768, 128), (257, 1)]:
        assert to_rgb8(Image.fromarray(np.full((4, 4), value, np.uint16))).getpixel((0, 0)) == (grey,) * 3
    assert to_rgb8(Image.fromarray(np.full((4, 4), 200, np.int32))).getpixel((0, 0)) == (200, 200, 200)  # 8-bit values
    assert to_rgb8(Image.fromarray(np.full((4, 4), 40000, np.int32))).getpixel((0, 0)) == (156, 156, 156)
    assert to_rgb8(Image.fromarray(np.full((4, 4), 0.5, np.float32))).getpixel((0, 0)) == (128, 128, 128)
    assert to_rgb8(Image.new("CMYK", (4, 4), (0, 0, 0, 0))).getpixel((0, 0)) == (255, 255, 255)

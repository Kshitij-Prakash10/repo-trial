"""Tests for the ML document scanner (synthetic data) and its HTTP API. Uses the committed model."""
from __future__ import annotations

import base64
import hashlib
import io
import os

import numpy as np
import pytest
from flask import Flask
from PIL import Image

from scanner.api import scan_bp
from scanner import geometry as geo
from scanner import dataset, infer, synth

HAS_MODEL = os.path.exists(infer.MODEL_PATH)
needs_model = pytest.mark.skipif(not HAS_MODEL, reason="models/scanner.joblib not trained")


@pytest.fixture(scope="module")
def sample() -> synth.Sample:
    return synth.make_sample(200_004)  # a held-out test seed


def test_generator_is_deterministic():
    a, b = synth.make_sample(5), synth.make_sample(5)
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))
    assert np.allclose(a.corners, b.corners) and np.array_equal(a.mask, b.mask)
    assert not np.array_equal(np.asarray(a.image), np.asarray(synth.make_sample(6).image))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_generator_ground_truth_is_sane(seed):
    s = synth.make_sample(seed)
    W, H = s.image.size
    assert 480 <= max(W, H) <= 1280 and s.image.mode == "RGB"
    assert s.mask.shape == (H, W) and s.mask.dtype == bool
    assert s.corners.shape == (4, 2) and geo.is_convex(s.corners)
    assert 0.15 <= s.mask.mean() <= 0.95
    # the mask is the polygon of the corners
    assert abs(s.mask.sum() - synth.poly_mask(s.corners, W, H).sum()) / s.mask.sum() < 0.02
    assert s.meta["data"] == "synthetic"
    # page centre is brighter than nothing in particular, but the mask centre must be inside the image
    cx, cy = s.corners.mean(0)
    assert s.mask[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))]


HAS_DATA = dataset.exists()


@pytest.mark.skipif(not HAS_DATA, reason="data/synthetic not generated")
def test_stored_dataset_matches_generator(tmp_path):
    """The committed synthetic data is what the generator writes, byte for byte (one file per split)."""
    for split, (seeds, style) in dataset.SPLITS.items():
        items = dataset.load_split(split)
        assert len(items) == len(seeds)
        assert {it["meta"]["seed"] for it in items} == set(seeds)
        row = dataset._write_one((seeds[3], style, str(tmp_path)))
        stored = items[3]
        assert stored["meta"]["seed"] == seeds[3]
        with open(stored["image"], "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == row["sha256"]
        assert np.allclose(stored["corners"].ravel(), [float(row[c]) for c in dataset.CORNER_COLS])


@pytest.mark.skipif(not HAS_DATA, reason="data/synthetic not generated")
def test_splits_are_disjoint():
    seeds = [{it["meta"]["seed"] for it in dataset.load_split(s)} for s in dataset.SPLITS]
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            assert not seeds[i] & seeds[j]


def test_ood_style_differs():
    s = synth.make_sample(1, "ood")
    assert s.meta["kind"] in synth.OOD_KINDS


def test_corner_error_is_order_invariant():
    q = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], float)
    assert geo.corner_error(np.roll(q, 1, 0), q) == pytest.approx(0)
    assert geo.corner_error(q + [1, 0], q) == pytest.approx(100 / np.hypot(10, 20))


@needs_model
def test_scan_finds_the_page(sample):
    res = infer.scan(sample.image, "gray")
    assert res["method"] == "ml" and res["page_found"]
    assert set(res) == {"page_found", "corners", "confidence", "image", "model_version", "method"}
    assert 0 <= res["confidence"] <= 1
    assert geo.corner_error(np.array(res["corners"]), sample.corners) < 3.0
    assert res["image"].mode == "L"
    w, h = res["image"].size
    assert 1.2 < max(w, h) / min(w, h) < 1.7


@needs_model
@pytest.mark.parametrize("mode,pil_mode", [("color", "RGB"), ("bw", "L")])
def test_scan_modes(sample, mode, pil_mode):
    res = infer.scan(sample.image, mode)
    assert res["image"].mode == pil_mode
    if mode == "bw":
        assert set(np.unique(np.asarray(res["image"]))) <= {0, 255}


def test_scan_rejects_bad_mode(sample):
    with pytest.raises(ValueError):
        infer.scan(sample.image, "sepia")


def test_fallback_without_model(monkeypatch, tmp_path, sample):
    monkeypatch.setattr(infer, "MODEL_PATH", str(tmp_path / "missing.joblib"))
    res = infer.scan(sample.image, "gray")
    W, H = sample.image.size
    assert res["method"] == "fallback" and res["page_found"] is False
    assert res["corners"] == [[0, 0], [W, 0], [W, H], [0, H]]
    assert res["image"].size == (W, H) and res["model_version"] == "none"


@pytest.mark.parametrize("img", [Image.new("RGB", (40, 30), (120, 120, 120)), Image.new("L", (3, 2), 255),
                                 Image.new("RGBA", (300, 200), (0, 0, 0, 0))])
def test_scan_never_raises_on_odd_images(img):
    res = infer.scan(img, "gray")
    assert res["method"] in ("ml", "fallback") and isinstance(res["image"], Image.Image)


# ---------------------------------------------------------------- no page: blank, noise, tiny

@needs_model
@pytest.mark.parametrize("img", [Image.new("RGB", (800, 600), "white"), Image.new("RGB", (1, 1), "white"),
                                 Image.new("RGB", (1, 1), "black"), Image.new("RGB", (31, 400), "white"),
                                 Image.new("RGB", (20, 20), (120, 120, 120))],
                         ids=["white-800x600", "white-1x1", "black-1x1", "31x400", "grey-20x20"])
def test_no_page_on_blank_or_tiny_images(img):
    res = infer.scan(img, "gray")
    assert res["page_found"] is False and res["method"] == "fallback"


@needs_model
@pytest.mark.parametrize("seed", [2026, 0, 1, 7, 12345])
def test_no_page_on_uniform_noise(seed):
    """Uniform random RGB noise used to give confidence ~0.63 and page_found=True."""
    img = Image.fromarray(np.random.default_rng(seed).integers(0, 256, (600, 800, 3), dtype=np.uint8))
    res = infer.scan(img, "gray")
    assert res["page_found"] is False and res["method"] == "fallback"


def test_tiny_image_is_rejected_before_the_quad_fit():
    prob = np.ones((8, 8))  # the classifier says "page" everywhere
    for size in ((1, 1), (31, 31), (31, 500)):
        res = infer.detect_from_prob(Image.new("RGB", size, "white"), prob, refine=False)
        assert res["page_found"] is False and res["confidence"] == 0.0


def test_noise_guard_keeps_a_real_page_and_rejects_noise(sample):
    """The guard needs inside == outside in colour AND texture AND no coherent step along the sides."""
    c = infer.coherence(sample.image, sample.corners)
    assert c["dprime"] > 0.2 and c["step_z"] > 5 and not infer.looks_like_noise(c)
    noise = Image.fromarray(np.random.default_rng(2026).integers(0, 256, (600, 800, 3), dtype=np.uint8))
    q = np.array([[200, 150], [600, 150], [600, 450], [200, 450]], float)
    c = infer.coherence(noise, q)
    assert c["dprime"] < infer.NOISE_MAX_DPRIME and infer.looks_like_noise(c)
    # a white page on the same noise: inside and outside differ, so it is kept
    a = np.asarray(noise).copy()
    a[150:450, 200:600] = 245
    assert not infer.looks_like_noise(infer.coherence(Image.fromarray(a), q))
    # a quad that fills the photo cannot be judged: never rejected
    full = np.array([[0, 0], [800, 0], [800, 600], [0, 600]], float)
    assert not infer.looks_like_noise(infer.coherence(noise, full))


# ---------------------------------------------------------------- image modes (scanner.convert)

def test_to_rgb_composites_transparency_on_white():
    from scanner.convert import to_rgb
    im = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    im.putpixel((1, 1), (10, 20, 30, 255))
    out = to_rgb(im)
    assert out.mode == "RGB" and out.getpixel((0, 0)) == (255, 255, 255) and out.getpixel((1, 1)) == (10, 20, 30)
    la = Image.new("LA", (2, 2), (0, 0))
    assert to_rgb(la).getpixel((0, 0)) == (255, 255, 255)
    p = Image.new("P", (2, 2), 0)
    p.info["transparency"] = 0
    assert to_rgb(p).getpixel((0, 0)) == (255, 255, 255)


def test_to_rgb_scales_16_bit_and_keeps_negative_floats():
    from scanner.convert import to_rgb
    a16 = np.tile(np.linspace(0, 65535, 256).astype(np.uint16), (4, 1))
    im16 = Image.fromarray(a16)
    assert im16.mode.startswith("I;16")
    out = np.asarray(to_rgb(im16))[0, :, 0]
    assert out[0] == 0 and out[-1] == 255 and 120 <= out[128] <= 135 and np.all(np.diff(out.astype(int)) >= 0)
    f = Image.fromarray(np.tile(np.linspace(-1, 1, 201).astype(np.float32), (4, 1)))
    assert f.mode == "F"
    out = np.asarray(to_rgb(f))[0, :, 0]
    assert out[0] == 0 and out[-1] == 255 and 126 <= out[100] <= 129  # -1 black, 0 mid grey, 1 white
    assert len(np.unique(out[:100])) > 50  # the negative half is not cut off at black
    f01 = Image.fromarray(np.full((2, 2), 0.5, np.float32))
    assert to_rgb(f01).getpixel((0, 0)) == (128, 128, 128)  # values up to 1.0 read as 0..1
    i8 = Image.fromarray(np.full((2, 2), 200, np.int32))
    assert i8.mode == "I" and to_rgb(i8).getpixel((0, 0)) == (200, 200, 200)
    rgb = Image.new("RGB", (2, 2), (1, 2, 3))
    assert to_rgb(rgb) is rgb and to_rgb(Image.new("L", (2, 2), 7)).getpixel((0, 0)) == (7, 7, 7)


@needs_model
def test_scan_keeps_transparent_and_16_bit_images_visible():
    rgba = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    for mode in ("color", "gray", "bw"):  # used to come out black
        assert np.asarray(infer.scan(rgba, mode)["image"]).min() == 255
    ramp = Image.fromarray(np.tile(np.linspace(0, 65535, 300).astype(np.uint16), (200, 1)))
    out = np.asarray(infer.scan(ramp, "gray")["image"])
    assert out.mean() > 100  # used to be all black (I;16 clamped by convert("RGB"))


def test_colour_enhance_leaves_a_blank_page_white():
    from scanner import enhance
    for mode in enhance.MODES:  # "color" used to turn a flat white image black
        assert np.asarray(enhance.enhance(Image.new("RGB", (60, 40), "white"), mode)).min() == 255


# ---------------------------------------------------------------- HTTP API

@pytest.fixture()
def client():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # lower than 10 MB on purpose; the blueprint lifts it for /api/scan
    app.register_blueprint(scan_bp)
    return app.test_client()


def _jpeg(img: Image.Image, **kw) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90, **kw)
    buf.seek(0)
    return buf


@needs_model
def test_api_happy_path(client, sample):
    r = client.post("/api/scan", data={"image": (_jpeg(sample.image), "page.jpg"), "mode": "gray"},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["page_found"] and j["method"] == "ml"
    assert len(j["corners"]) == 4 and len(j["corners_norm"]) == 4
    assert all(-0.3 <= v <= 1.3 for p in j["corners_norm"] for v in p)
    W, H = sample.image.size
    assert geo.corner_error(np.array(j["corners"]), sample.corners) < 3.0
    out = Image.open(io.BytesIO(base64.b64decode(j["image_png_b64"])))
    assert out.format == "PNG" and out.size == (j["width"], j["height"])
    assert j["elapsed_ms"] > 0 and isinstance(j["model_version"], str)


@needs_model
def test_api_applies_exif_orientation(client, sample):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise for display
    stored = sample.image.transpose(Image.Transpose.ROTATE_90)  # what a phone stores for that tag
    r = client.post("/api/scan", data={"image": (_jpeg(stored, exif=exif), "p.jpg")},
                    content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200 and j["page_found"]
    assert geo.corner_error(np.array(j["corners"]), sample.corners) < 3.0


def test_api_missing_field(client):
    r = client.post("/api/scan", data={"mode": "gray"}, content_type="multipart/form-data")
    assert r.status_code == 400 and "error" in r.get_json()


def test_api_not_an_image(client):
    r = client.post("/api/scan", data={"image": (io.BytesIO(b"%PDF-1.4 not an image"), "x.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "error" in r.get_json()


def test_api_bad_mode(client, sample):
    r = client.post("/api/scan", data={"image": (_jpeg(sample.image), "p.jpg"), "mode": "sepia"},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "mode" in r.get_json()["error"]


def test_api_too_large(client):
    big = io.BytesIO(b"\xff\xd8\xff" + b"0" * (10 * 1024 * 1024 + 10))
    r = client.post("/api/scan", data={"image": (big, "big.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 413 and "error" in r.get_json()


def test_api_accepts_upload_over_app_default_limit(client):
    """A 3 MB PNG is under the 10 MB scan limit even though the app default is 2 MB."""
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (1100, 1000, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=0)
    assert buf.tell() > 2 * 1024 * 1024
    buf.seek(0)
    r = client.post("/api/scan", data={"image": (buf, "noise.png"), "mode": "bw"}, content_type="multipart/form-data")
    assert r.status_code == 200


def test_api_status(client):
    j = client.get("/api/scan/status").get_json()
    assert set(j) == {"available", "model_version", "metrics"}
    assert j["available"] == HAS_MODEL
    if HAS_MODEL:
        assert j["metrics"]["data"].startswith("synthetic")

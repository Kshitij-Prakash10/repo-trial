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

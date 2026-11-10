"""Tests for ocr/api.py: status, the unavailable path, bad input, and one real end-to-end read.

Everything except the last test runs without torch / docTR: missing dependencies are simulated, and the
response shape is checked with a stand-in pipeline. All images are generated here (synthetic).
"""
from __future__ import annotations

import io
import subprocess
import sys
import threading

import pytest
from PIL import Image, ImageDraw, ImageFont

import server
from ocr import api


@pytest.fixture()
def client(monkeypatch):
    """App with the OCR blueprint and a clean module state (no pipeline loaded, no remembered error)."""
    monkeypatch.setattr(api, "_pipeline", None)
    monkeypatch.setattr(api, "_load_error", None)
    app = server.create_app(with_scanner=False)
    app.config["TESTING"] = True
    assert app.config["OCR_REGISTERED"] is True
    return app.test_client()


def png_bytes(size=(80, 60), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, fmt)
    return buf.getvalue()


def post(client, data: bytes, name: str = "page.png", **fields):
    return client.post("/api/ocr", data={"image": (io.BytesIO(data), name), **fields},
                       content_type="multipart/form-data")


class FakePipeline:
    """Stands in for ocr.pipeline.OcrPipeline: same result keys, no models."""

    name = {"detector": "pretrained", "recogniser": "finetuned"}

    def __init__(self):
        self.use_scanner = True
        self.calls = []

    def run(self, image):
        self.calls.append({"size": image.size, "use_scanner": self.use_scanner})
        words = [{"text": "Total", "confidence": 0.9, "detection_score": 0.8, "box": [1.0, 2.0, 30.0, 12.0],
                  "source_quad": [[1, 2], [30, 2], [30, 12], [1, 12]], "line": 0},
                 {"text": "9.00", "confidence": 0.7, "detection_score": 0.8, "box": [35.0, 2.0, 60.0, 12.0],
                  "source_quad": [[35, 2], [60, 2], [60, 12], [35, 12]], "line": 0}]
        return {"text": "Total 9.00", "words": words,
                "lines": [{"text": "Total 9.00", "words": [0, 1], "box": [1.0, 2.0, 60.0, 12.0], "confidence": 0.8}],
                "page": {"width": image.size[0], "height": image.size[1], "is_scanner_crop": False},
                "source": {"width": image.size[0], "height": image.size[1]},
                "scanner": {"used": False, "reason": "scanner stage disabled"}, "models": self.name, "seconds": 0.01}


# ---------------------------------------------------------------- status

def test_status_reports_available_or_a_reason(client):
    r = client.get("/api/ocr/status")
    assert r.status_code == 200 and r.is_json
    body = r.get_json()
    assert isinstance(body["available"], bool)
    assert r.headers["Cache-Control"] == "no-store"
    if body["available"]:
        assert set(body["models"]) == {"detector", "recogniser"}
        assert set(body["models"].values()) <= {"pretrained", "finetuned"}
    else:
        assert isinstance(body["reason"], str) and body["reason"]


def test_status_and_post_when_packages_are_missing(client, monkeypatch):
    monkeypatch.setattr(api, "PACKAGES", ("a_package_that_is_not_installed_xyz",))
    body = client.get("/api/ocr/status").get_json()
    assert body["available"] is False
    assert "requirements-ocr.txt" in body["reason"]
    assert "models" not in body
    r = post(client, png_bytes())
    assert r.status_code == 503 and r.is_json
    assert "requirements-ocr.txt" in r.get_json()["error"]


def test_status_when_weights_are_not_cached(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "PACKAGES", ("json",))  # installed everywhere: pretend the packages are there
    monkeypatch.setenv("DOCTR_CACHE_DIR", str(tmp_path))
    body = client.get("/api/ocr/status").get_json()
    assert body["available"] is False and "model files" in body["reason"]
    assert post(client, png_bytes()).status_code == 503  # and nothing is downloaded
    models = tmp_path / "models"
    models.mkdir()
    (models / "fast_base-00000000.pt").write_bytes(b"x")
    (models / "crnn_mobilenet_v3_large_pt-00000000.pt").write_bytes(b"x")
    assert client.get("/api/ocr/status").get_json()["available"] is True


def test_load_failure_is_503_and_shows_in_status(client, monkeypatch):
    monkeypatch.setattr(api, "missing_packages", lambda: [])
    monkeypatch.setattr(api, "missing_weights", lambda: [])
    monkeypatch.setitem(sys.modules, "ocr.pipeline", None)  # makes "from ocr.pipeline import ..." raise
    r = post(client, png_bytes())
    assert r.status_code == 503 and "could not be loaded" in r.get_json()["error"]
    body = client.get("/api/ocr/status").get_json()
    assert body["available"] is False and "could not be loaded" in body["reason"]


def test_ocr_without_blueprint_is_json_404():
    app = server.create_app(with_scanner=False, with_ocr=False)
    c = app.test_client()
    for r in (c.get("/api/ocr/status"), c.post("/api/ocr")):
        assert r.status_code == 404 and r.is_json
        assert "text" in r.get_json()["error"].lower()


def test_server_start_and_status_do_not_import_torch():
    code = ("import sys, server\n"
            "c = server.create_app().test_client()\n"
            "assert c.get('/api/ocr/status').status_code == 200\n"
            "print('torch' in sys.modules or 'doctr' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=api.ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"


# ---------------------------------------------------------------- bad input

def test_missing_image_field_is_400(client):
    r = client.post("/api/ocr", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and "image" in r.get_json()["error"]


@pytest.mark.parametrize("data, message", [
    (b"", "empty"),
    (b"this is not an image at all", "not a readable image"),
    (b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\n", "not a readable image"),
    (png_bytes()[:40], "not a readable image"),  # truncated
    (png_bytes(fmt="GIF"), "JPEG, PNG or WebP"),
])
def test_bad_image_is_400(client, monkeypatch, data, message):
    monkeypatch.setattr(api, "_pipeline", FakePipeline())  # a bad file must be refused before any reading
    r = post(client, data)
    assert r.status_code == 400 and r.is_json
    assert message in r.get_json()["error"]
    assert api._pipeline.calls == []


def test_bad_scanner_flag_is_400(client):
    r = post(client, png_bytes(), scanner="maybe")
    assert r.status_code == 400 and "scanner" in r.get_json()["error"]


def test_too_large_is_413(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_BYTES", 1000)
    r = post(client, b"x" * 2000)
    assert r.status_code == 413 and "larger than" in r.get_json()["error"]


def test_get_is_405_json(client):
    r = client.get("/api/ocr")
    assert r.status_code == 405 and r.is_json


# ---------------------------------------------------------------- response shape (stand-in pipeline)

def test_response_shape_and_scanner_flag(client, monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(api, "_pipeline", fake)
    r = post(client, png_bytes((120, 90)))
    assert r.status_code == 200
    body = r.get_json()
    assert body["text"] == "Total 9.00"
    assert body["lines"] == [{"text": "Total 9.00", "box": [1.0, 2.0, 60.0, 12.0], "confidence": 0.8}]
    assert body["words"][0] == {"text": "Total", "confidence": 0.9, "box": [1.0, 2.0, 30.0, 12.0], "line": 0}
    assert body["page"] == {"width": 120, "height": 90}
    assert body["seconds"] == 0.01
    assert fake.calls == [{"size": (120, 90), "use_scanner": False}]  # the scanner stage is off by default
    assert post(client, png_bytes(), scanner="1").status_code == 200
    assert fake.calls[-1]["use_scanner"] is True
    assert post(client, png_bytes(fmt="JPEG"), "page.jpg", scanner="0").status_code == 200
    assert fake.calls[-1]["use_scanner"] is False
    status = client.get("/api/ocr/status").get_json()
    if status["available"]:
        assert status["models"] == fake.name


def test_one_page_is_read_at_a_time(monkeypatch):
    active, most = [0], [0]
    gate = threading.Lock()

    class Slow(FakePipeline):
        def run(self, image):
            with gate:
                active[0] += 1
                most[0] = max(most[0], active[0])
            threading.Event().wait(0.05)
            with gate:
                active[0] -= 1
            return super().run(image)

    monkeypatch.setattr(api, "_pipeline", Slow())
    page = Image.new("RGB", (40, 30), "white")
    threads = [threading.Thread(target=api.read_page, args=(page,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert most[0] == 1


# ---------------------------------------------------------------- real models

def test_end_to_end_on_a_generated_page(client):
    pytest.importorskip("torch")
    pytest.importorskip("doctr")
    if api.missing_weights():
        pytest.skip("docTR pretrained weights are not in the local cache")
    page = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=44)
    for i, line in enumerate(["CITY CLINIC PHARMACY", "PARACETAMOL 500 MG", "TOTAL 12.50"]):
        draw.text((60, 50 + i * 110), line, fill="black", font=font)
    buf = io.BytesIO()
    page.save(buf, "PNG")
    assert client.get("/api/ocr/status").get_json()["available"] is True
    r = post(client, buf.getvalue())
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["page"] == {"width": 900, "height": 420}
    assert len(body["lines"]) == 3 and len(body["words"]) >= 5
    text = body["text"].upper()
    for word in ("CLINIC", "PHARMACY", "PARACETAMOL", "TOTAL"):
        assert word in text, body["text"]
    for w in body["words"]:
        assert set(w) == {"text", "confidence", "box", "line"}
        assert 0.0 <= w["confidence"] <= 1.0 and len(w["box"]) == 4
        assert 0 <= w["line"] < len(body["lines"])
    for ln in body["lines"]:
        assert set(ln) == {"text", "box", "confidence"}
    assert body["seconds"] > 0
    assert client.get("/api/ocr/status").get_json()["loaded"] is True

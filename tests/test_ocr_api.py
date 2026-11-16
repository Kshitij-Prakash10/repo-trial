"""Tests for ocr/api.py: status, the unavailable path, bad input, the two modes, and real end-to-end reads.

Everything except the last two tests runs without torch / docTR / transformers: missing dependencies are
simulated, and the response shape and the choice between the handwriting mode (ocr.scan.Scanner) and the
printed-only pipeline are checked with stand-ins. All images are generated here (synthetic).
"""
from __future__ import annotations

import io
import subprocess
import sys
import threading
import types

import pytest
from PIL import Image, ImageDraw, ImageFont

import server
from ocr import api


@pytest.fixture()
def client(monkeypatch):
    """App with the OCR blueprint and a clean module state (no models loaded, no remembered error)."""
    monkeypatch.setattr(api, "_scanner", None)
    monkeypatch.setattr(api, "_pipeline", None)
    monkeypatch.setattr(api, "_load_error", None)
    monkeypatch.setattr(api, "_hw_load_error", None)
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


class FakeScanner:
    """Stands in for ocr.scan.Scanner: same result keys, no models. The second line has no text and is
    dropped, as the real Scanner does, so the words' line numbers must be remapped by the API."""

    def __init__(self):
        self.ocr = FakePipeline()
        self.calls = []

    def run(self, image):
        self.calls.append({"size": image.size, "use_scanner": self.ocr.use_scanner})
        box = [1.0, 2.0, 30.0, 12.0]
        words = [{"text": "Total", "confidence": 0.9, "detection_score": 0.8, "box": box, "line": 0},
                 {"text": "x", "confidence": 0.2, "detection_score": 0.5, "box": box, "line": 1},
                 {"text": "tak", "confidence": 0.3, "detection_score": 0.7, "box": box, "line": 2},
                 {"text": "dally", "confidence": 0.4, "detection_score": 0.7, "box": box, "line": 2}]
        lines = [{"text": "Total", "box": box, "confidence": 0.9, "script": "printed", "handwritten_share": 0.0,
                  "words": [0], "low_confidence": False},
                 # (the line of word 1 had no text and was dropped)
                 {"text": "take daily", "box": box, "confidence": 0.45, "script": "handwritten",
                  "handwritten_share": 0.9, "words": [2, 3], "low_confidence": True,
                  "handwriting_text": "take daily", "handwriting_confidence": 0.45,
                  "printed_text": "tak dally", "printed_confidence": 0.35}]
        return {"text": "Total\ntake daily", "confidences": [0.9, 0.45], "lines": lines, "words": words,
                "preprocessing": {"quarter_turns": 1, "orientation_confidence": 0.97, "deskew_degrees": 0.0,
                                  "flip_check": {"margin": 0.4, "crops": 3, "flipped": False}},
                "page": {"width": image.size[1], "height": image.size[0], "is_scanner_crop": False},
                "source": {"width": image.size[1], "height": image.size[0]},
                "scanner": {"used": False, "reason": "scanner stage disabled"},
                "models": {**self.ocr.name, "handwriting": "/some/where/checkpoints/handwriting_v1"}, "seconds": 0.02}


def pretend_installed(monkeypatch, handwriting: bool = True):
    """Make the availability checks pass without the real packages or model files."""
    monkeypatch.setattr(api, "PACKAGES", ("json",))  # installed everywhere
    monkeypatch.setattr(api, "missing_weights", lambda: [])
    monkeypatch.setattr(api, "HW_PACKAGES", ("json",) if handwriting else ("a_package_that_is_not_installed_xyz",))
    monkeypatch.setattr(api, "missing_handwriting_files", lambda: [])
    # "from ocr.pipeline import OcrPipeline" then gives a stand-in whose load() needs no models
    fake = types.ModuleType("ocr.pipeline")
    fake.OcrPipeline = types.SimpleNamespace(load=lambda weights, use_scanner=True: FakePipeline())
    monkeypatch.setitem(sys.modules, "ocr.pipeline", fake)


# ---------------------------------------------------------------- status

def test_status_reports_available_or_a_reason(client):
    r = client.get("/api/ocr/status")
    assert r.status_code == 200 and r.is_json
    body = r.get_json()
    assert isinstance(body["available"], bool)
    assert isinstance(body["handwriting"], bool)
    assert r.headers["Cache-Control"] == "no-store"
    if body["available"]:
        assert set(body["models"]) == {"detector", "recogniser"}
        assert set(body["models"].values()) <= {"pretrained", "finetuned"}
        assert body["loaded"] is False
        if not body["handwriting"]:
            assert isinstance(body["handwriting_reason"], str) and body["handwriting_reason"]
    else:
        assert isinstance(body["reason"], str) and body["reason"]
        assert body["handwriting"] is False


def test_status_and_post_when_packages_are_missing(client, monkeypatch):
    monkeypatch.setattr(api, "PACKAGES", ("a_package_that_is_not_installed_xyz",))
    body = client.get("/api/ocr/status").get_json()
    assert body["available"] is False and body["handwriting"] is False
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
    monkeypatch.setattr(api, "PACKAGES", ("json",))
    monkeypatch.setattr(api, "HW_PACKAGES", ("a_package_that_is_not_installed_xyz",))
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
            "print('torch' in sys.modules or 'doctr' in sys.modules or 'transformers' in sys.modules)\n")
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
    assert set(body) == {"text", "lines", "words", "page", "seconds", "models", "handwriting", "preprocessing"}
    assert body["lines"] == [{"text": "Total 9.00", "box": [1.0, 2.0, 60.0, 12.0], "confidence": 0.8,
                              "script": "printed", "low_confidence": False}]
    assert body["words"][0] == {"text": "Total", "confidence": 0.9, "box": [1.0, 2.0, 30.0, 12.0], "line": 0}
    assert body["page"] == {"width": 120, "height": 90}
    assert body["seconds"] == 0.01
    assert body["models"] == fake.name
    assert body["handwriting"] is False and body["preprocessing"] is None  # the printed-only pipeline
    assert fake.calls == [{"size": (120, 90), "use_scanner": False}]  # the scanner stage is off by default
    assert post(client, png_bytes(), scanner="1").status_code == 200
    assert fake.calls[-1]["use_scanner"] is True
    assert post(client, png_bytes(fmt="JPEG"), "page.jpg", scanner="0").status_code == 200
    assert fake.calls[-1]["use_scanner"] is False
    status = client.get("/api/ocr/status").get_json()
    if status["available"]:
        assert status["models"] == fake.name
        assert status["handwriting"] is False and status["handwriting_reason"]


def test_printed_line_below_the_limit_is_flagged(client, monkeypatch):
    class Unsure(FakePipeline):
        def run(self, image):
            res = super().run(image)
            res["lines"][0]["confidence"] = 0.31
            return res

    monkeypatch.setattr(api, "_pipeline", Unsure())
    line = post(client, png_bytes()).get_json()["lines"][0]
    assert line["low_confidence"] is True and line["script"] == "printed"


def test_exif_rotation_is_applied_before_reading(client, monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(api, "_pipeline", fake)
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: the camera was turned; viewers show the picture 60 x 80
    buf = io.BytesIO()
    Image.new("RGB", (80, 60), "white").save(buf, "JPEG", exif=exif)
    assert post(client, buf.getvalue(), "photo.jpg").status_code == 200
    assert fake.calls[-1]["size"] == (60, 80)


# ---------------------------------------------------------------- handwriting mode or printed-only (stand-ins)

def test_handwriting_mode_response_and_status(client, monkeypatch):
    pretend_installed(monkeypatch)
    fake = FakeScanner()
    loads = []
    monkeypatch.setattr(api, "_load_scanner", lambda weights: loads.append(weights) or fake)
    status = client.get("/api/ocr/status").get_json()
    assert status["available"] is True and status["handwriting"] is True and status["loaded"] is False
    assert "handwriting_reason" not in status and loads == []  # the status loads nothing
    r = post(client, png_bytes((120, 90)))
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert set(body) == {"text", "lines", "words", "page", "seconds", "models", "handwriting", "preprocessing"}
    assert body["handwriting"] is True and body["text"] == "Total\ntake daily"
    assert body["lines"] == [
        {"text": "Total", "box": [1.0, 2.0, 30.0, 12.0], "confidence": 0.9, "script": "printed", "low_confidence": False},
        {"text": "take daily", "box": [1.0, 2.0, 30.0, 12.0], "confidence": 0.45, "script": "handwritten",
         "low_confidence": True}]
    assert [w["line"] for w in body["words"]] == [0, None, 1, 1]  # indices into the lines of the answer
    assert all(set(w) == {"text", "confidence", "box", "line"} for w in body["words"])
    assert body["preprocessing"]["quarter_turns"] == 1 and body["preprocessing"]["orientation_confidence"] == 0.97
    assert body["page"] == {"width": 90, "height": 120}  # the page as read, after the turn
    assert body["models"] == {"detector": "pretrained", "recogniser": "finetuned", "handwriting": "handwriting_v1"}
    assert fake.calls == [{"size": (120, 90), "use_scanner": False}]
    assert post(client, png_bytes(), scanner="1").status_code == 200
    assert fake.calls[-1]["use_scanner"] is True
    assert len(loads) == 1 and loads[0] in ("finetuned", "pretrained")  # loaded once, on the first POST
    status = client.get("/api/ocr/status").get_json()
    assert status["handwriting"] is True and status["loaded"] is True and status["models"] == fake.ocr.name


def test_printed_only_without_the_handwriting_packages(client, monkeypatch):
    pretend_installed(monkeypatch, handwriting=False)
    monkeypatch.setattr(api, "_load_scanner", lambda weights: pytest.fail("the Scanner must not be loaded"))
    status = client.get("/api/ocr/status").get_json()
    assert status["available"] is True and status["handwriting"] is False
    assert "requirements-handwriting.txt" in status["handwriting_reason"]
    r = post(client, png_bytes())
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["text"] == "Total 9.00" and body["handwriting"] is False and body["preprocessing"] is None
    assert body["lines"][0]["script"] == "printed"
    status = client.get("/api/ocr/status").get_json()
    assert status["loaded"] is True and status["handwriting"] is False
    assert "requirements-handwriting.txt" in status["handwriting_reason"]


def test_printed_only_when_a_handwriting_file_is_missing(client, monkeypatch):
    pretend_installed(monkeypatch)
    monkeypatch.setattr(api, "missing_handwriting_files", lambda: ["checkpoints/orientation_v1/orientation.pt"])
    monkeypatch.setattr(api, "_load_scanner", lambda weights: pytest.fail("the Scanner must not be loaded"))
    status = client.get("/api/ocr/status").get_json()
    assert status["available"] is True and status["handwriting"] is False
    assert "orientation_v1" in status["handwriting_reason"]
    assert post(client, png_bytes()).get_json()["handwriting"] is False


def test_missing_handwriting_files_lists_what_is_not_on_disk(monkeypatch, tmp_path):
    hw = tmp_path / "checkpoints" / "handwriting_v1"
    hw.mkdir(parents=True)
    present = hw / "config.json"
    present.write_text("{}")
    monkeypatch.setattr(api, "ROOT", str(tmp_path))
    monkeypatch.setattr(api, "HW_DIR", str(hw))
    monkeypatch.setattr(api, "HW_FILES", (str(present), str(tmp_path / "checkpoints" / "scriptclf_v1" / "scriptclf.pt")))
    missing = [m.replace("\\", "/") for m in api.missing_handwriting_files()]
    assert missing == ["checkpoints/scriptclf_v1/scriptclf.pt", "checkpoints/handwriting_v1/model.safetensors"]
    (hw / "model-00001-of-00002.safetensors").write_bytes(b"x")  # sharded weights count
    assert len(api.missing_handwriting_files()) == 1


def test_scanner_load_failure_falls_back_to_printed(client, monkeypatch):
    pretend_installed(monkeypatch)

    def broken(weights):
        raise RuntimeError("unreadable checkpoint")

    monkeypatch.setattr(api, "_load_scanner", broken)
    assert client.get("/api/ocr/status").get_json()["handwriting"] is True  # nothing was tried yet
    r = post(client, png_bytes())
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["handwriting"] is False and r.get_json()["text"] == "Total 9.00"
    status = client.get("/api/ocr/status").get_json()
    assert status["available"] is True and status["handwriting"] is False
    assert "could not be loaded (RuntimeError)" in status["handwriting_reason"]


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

    monkeypatch.setattr(api, "_scanner", None)
    monkeypatch.setattr(api, "_pipeline", Slow())
    page = Image.new("RGB", (40, 30), "white")
    threads = [threading.Thread(target=api.read_page, args=(page,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert most[0] == 1


# ---------------------------------------------------------------- real models

_real = {}  # the models loaded by the first real test, reused by the next (loading takes seconds)


def real_models(monkeypatch):
    """Skip without the packages or weights; otherwise reuse the models an earlier real test loaded."""
    pytest.importorskip("torch")
    pytest.importorskip("doctr")
    if api.missing_weights():
        pytest.skip("docTR pretrained weights are not in the local cache")
    for name, value in _real.items():
        monkeypatch.setattr(api, name, value)


def keep_real_models():
    _real.update(_scanner=api._scanner, _pipeline=api._pipeline, _hw_load_error=api._hw_load_error)


def generated_page() -> Image.Image:
    page = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=44)
    for i, line in enumerate(["CITY CLINIC PHARMACY", "PARACETAMOL 500 MG", "TOTAL 12.50"]):
        draw.text((60, 50 + i * 110), line, fill="black", font=font)
    return page


def test_end_to_end_on_a_generated_page(client, monkeypatch):
    real_models(monkeypatch)
    buf = io.BytesIO()
    generated_page().save(buf, "PNG")
    before = client.get("/api/ocr/status").get_json()
    assert before["available"] is True
    r = post(client, buf.getvalue())
    keep_real_models()
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
        assert set(ln) == {"text", "box", "confidence", "script", "low_confidence"}
        assert ln["script"] == "printed" and ln["low_confidence"] is False
    assert body["seconds"] > 0
    after = client.get("/api/ocr/status").get_json()
    assert after["loaded"] is True
    # the mode the status announced is the mode that read the page
    assert body["handwriting"] is before["handwriting"] is after["handwriting"]
    if body["handwriting"]:
        assert body["preprocessing"]["quarter_turns"] == 0
        assert body["models"]["handwriting"] == "handwriting_v1"
    else:
        assert body["preprocessing"] is None


def test_end_to_end_turned_page_in_handwriting_mode(client, monkeypatch):
    """A generated printed page turned by 90 degrees is put upright and read (real Scanner)."""
    real_models(monkeypatch)
    pytest.importorskip("transformers")
    if not client.get("/api/ocr/status").get_json()["handwriting"]:
        pytest.skip("the handwriting mode is not usable here (packages or checkpoints missing)")
    buf = io.BytesIO()
    generated_page().rotate(90, expand=True).save(buf, "PNG")
    r = post(client, buf.getvalue())
    keep_real_models()
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["handwriting"] is True
    assert body["preprocessing"]["quarter_turns"] in (1, 3), body["preprocessing"]
    assert body["page"] == {"width": 900, "height": 420}  # the upright page
    text = body["text"].upper()
    for word in ("CLINIC", "PHARMACY", "PARACETAMOL", "TOTAL"):
        assert word in text, body["text"]
    for w in body["words"]:
        assert w["line"] is None or 0 <= w["line"] < len(body["lines"])

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
    monkeypatch.setattr(api, "_v2_error", None)
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
    """Stands in for ocr.scan.Scanner (the v2 pipeline): same attributes and result keys, no models."""

    def __init__(self):
        self.ocr = FakePipeline()  # the printed pipeline inside, which holds use_scanner and name
        self.do_deskew = False
        self.orient = object()
        self.hw = object()
        self.calls = []
        self.turns = 0

    def run(self, image):
        self.calls.append({"size": image.size, "mode": image.mode, "use_scanner": self.ocr.use_scanner,
                           "deskew": self.do_deskew})
        base = self.ocr.run(image)
        lines = [{"text": "Total 9.00", "box": [1.0, 2.0, 60.0, 12.0], "confidence": 0.8, "script": "printed",
                  "handwritten_share": 0.02, "words": [0, 1], "low_confidence": False, "confirm_numbers": True},
                 {"text": "Amoxicillin 25 mg", "box": [1.0, 20.0, 70.0, 34.0], "confidence": 0.97,
                  "script": "handwritten", "handwritten_share": 0.9, "words": [], "handwriting_text": "Amoxicillin 25 mg",
                  "handwriting_confidence": 0.97, "printed_text": "Amoxicilin Z5 mq", "printed_confidence": 0.41,
                  "low_confidence": False, "confirm_numbers": True},
                 {"text": "see overleaf", "box": [1.0, 40.0, 50.0, 52.0], "confidence": 0.31, "script": "handwritten",
                  "handwritten_share": 0.8, "words": [], "low_confidence": True, "confirm_numbers": False}]
        pre = {"quarter_turns": self.turns, "orientation_confidence": 0.9876,
               "deskew_degrees": 1.25 if self.do_deskew else 0.0,
               "flip_check": {"margin": 0.31, "crops": 2, "flipped": False}}
        return {"text": "\n".join(ln["text"] for ln in lines), "confidences": [ln["confidence"] for ln in lines],
                "lines": lines, "words": base["words"], "preprocessing": pre, "page": base["page"],
                "source": base["source"], "scanner": base["scanner"],
                "models": {**self.ocr.name, "handwriting": "checkpoints/handwriting_v1"}, "seconds": 0.02}


def v2_ready(monkeypatch, tmp_path, packages=("json",), missing=()):
    """Make the v2 capability check see its packages and its three files (minus `missing`)."""
    files = {}
    for stage in api.V2_FILES:
        f = tmp_path / (stage.replace(" / ", "_").replace(" ", "_") + ".bin")
        if stage not in missing:
            f.write_bytes(b"x")
        files[stage] = str(f)
    monkeypatch.setattr(api, "V2_FILES", files)
    monkeypatch.setattr(api, "V2_PACKAGES", tuple(packages))


def v1_ready(monkeypatch):
    monkeypatch.setattr(api, "missing_packages", lambda: [])
    monkeypatch.setattr(api, "missing_weights", lambda: [])


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
    # no fine-tuned checkpoint for either stage: both then need docTR's pretrained file
    monkeypatch.setattr(api, "DET_CKPT", str(tmp_path / "no_det.pt"))
    monkeypatch.setattr(api, "REC_CKPT", str(tmp_path / "no_rec.pt"))
    assert api.missing_weights() == ["detector", "recogniser"]
    body = client.get("/api/ocr/status").get_json()
    assert body["available"] is False and "model files" in body["reason"]
    assert "detector and recogniser" in body["reason"]
    assert post(client, png_bytes()).status_code == 503  # and nothing is downloaded
    models = tmp_path / "models"
    models.mkdir()
    (models / "fast_base-00000000.pt").write_bytes(b"x")
    assert api.missing_weights() == ["recogniser"]
    assert client.get("/api/ocr/status").get_json()["available"] is False
    (models / "crnn_mobilenet_v3_large_pt-00000000.pt").write_bytes(b"x")
    assert api.missing_weights() == []
    assert client.get("/api/ocr/status").get_json()["available"] is True


def test_a_finetuned_checkpoint_needs_no_cached_weights(client, monkeypatch, tmp_path):
    """The recogniser's fine-tuned checkpoint is in the repository, so only the detector's pretrained
    file has to be in the docTR cache."""
    monkeypatch.setattr(api, "PACKAGES", ("json",))
    monkeypatch.setenv("DOCTR_CACHE_DIR", str(tmp_path))  # an empty cache
    rec = tmp_path / "rec_ft.pt"
    rec.write_bytes(b"x")
    monkeypatch.setattr(api, "REC_CKPT", str(rec))
    monkeypatch.setattr(api, "DET_CKPT", str(tmp_path / "no_det.pt"))
    assert api.missing_weights() == ["detector"]
    body = client.get("/api/ocr/status").get_json()
    assert body["available"] is False and "(detector)" in body["reason"] and "recogniser" not in body["reason"]
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "fast_base-00000000.pt").write_bytes(b"x")
    assert api.missing_weights() == []
    assert client.get("/api/ocr/status").get_json()["available"] is True
    det = tmp_path / "det_ft.pt"  # with both fine-tuned checkpoints the cache is not looked at
    det.write_bytes(b"x")
    monkeypatch.setattr(api, "DET_CKPT", str(det))
    monkeypatch.setenv("DOCTR_CACHE_DIR", str(tmp_path / "nowhere"))
    assert api.missing_weights() == []


def test_the_repository_recogniser_checkpoint_is_found():
    import os
    assert os.path.exists(api.REC_CKPT)  # shipped in models/ocr
    assert "recogniser" not in api.missing_weights()


@pytest.fixture
def unavailable_loaders(monkeypatch):
    v1_ready(monkeypatch)
    monkeypatch.setattr(api, "v2_problem", lambda: None)

    def broken():
        raise OSError("unreadable checkpoint")

    monkeypatch.setattr(api, "_load_v2", broken)
    monkeypatch.setattr(api, "_load_v1", broken)


def test_load_failure_is_503_and_shows_in_status(client, unavailable_loaders):
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
    # the three keys of the first version, plus the flags (v1 defaults: nothing flagged, all printed)
    assert body["lines"] == [{"text": "Total 9.00", "box": [1.0, 2.0, 60.0, 12.0], "confidence": 0.8,
                              "low_confidence": False, "confirm_numbers": True, "script": "printed", "has_numbers": True}]
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
        assert set(ln) == {"text", "box", "confidence", "low_confidence", "confirm_numbers", "script", "has_numbers"}
    assert body["seconds"] > 0
    assert client.get("/api/ocr/status").get_json()["loaded"] is True


# ---------------------------------------------------------------- limits added after the review

def test_unavailable_is_answered_before_the_image_is_decoded(client, monkeypatch):
    monkeypatch.setattr(api, "missing_packages", lambda: ["torch"])

    def no_decode(*a, **k):
        raise AssertionError("the image must not be decoded when OCR cannot run")

    monkeypatch.setattr(api.ImageOps, "exif_transpose", no_decode)
    assert post(client, png_bytes()).status_code == 503


@pytest.mark.parametrize("size", [(30000, 100), (13000, 1000), (6000, 5000), (4000, 100)])
def test_oversized_or_extreme_images_are_400(client, monkeypatch, size):
    monkeypatch.setattr(api, "_pipeline", FakePipeline())
    buf = io.BytesIO()
    Image.new("L", size, 255).save(buf, "PNG", compress_level=9)
    r = post(client, buf.getvalue())
    assert r.status_code == 400 and ("too many pixels" in r.get_json()["error"] or "too narrow" in r.get_json()["error"])


def test_busy_is_503_with_retry_after_when_too_many_wait(client, monkeypatch):
    monkeypatch.setattr(api, "_pipeline", FakePipeline())
    monkeypatch.setattr(api, "_slots", threading.BoundedSemaphore(1))
    assert api._slots.acquire(blocking=False)  # another request is inside
    r = post(client, png_bytes())
    assert r.status_code == 503 and r.headers["Retry-After"] and "Try again" in r.get_json()["error"]
    api._slots.release()
    assert post(client, png_bytes()).status_code == 200  # and the slot was given back


def test_waiting_for_a_turn_times_out_as_busy(client, monkeypatch):
    monkeypatch.setattr(api, "_pipeline", FakePipeline())
    monkeypatch.setattr(api, "LOCK_WAIT", 0.05)
    lock = threading.Lock()
    monkeypatch.setattr(api, "_lock", lock)
    lock.acquire()  # a page that never finishes
    try:
        r = post(client, png_bytes())
    finally:
        lock.release()
    assert r.status_code == 503 and r.headers["Retry-After"]
    assert post(client, png_bytes()).status_code == 200


def test_a_load_failure_is_retried_after_a_while(client, monkeypatch, unavailable_loaders):
    assert post(client, png_bytes()).status_code == 503
    assert client.get("/api/ocr/status").get_json()["available"] is False
    monkeypatch.setattr(api, "_load_error_at", api.time.monotonic() - api.LOAD_RETRY - 1)  # time passes
    assert client.get("/api/ocr/status").get_json()["available"] is True
    monkeypatch.setattr(api, "_load_v1", FakePipeline)
    assert post(client, png_bytes()).status_code == 200


# ---------------------------------------------------------------- v2 pipeline (ocr.scan.Scanner) and the v1 fallback

def test_v2_flags_and_preprocessing_reach_the_response(client, monkeypatch):
    fake = FakeScanner()
    fake.turns = 3
    monkeypatch.setattr(api, "_pipeline", fake)
    r = post(client, png_bytes((120, 90)))
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert set(body) == {"text", "lines", "words", "page", "seconds", "models", "pipeline", "preprocessing"}
    assert body["pipeline"] == "v2"
    assert body["text"] == "Total 9.00\nAmoxicillin 25 mg\nsee overleaf"
    assert body["lines"] == [
        {"text": "Total 9.00", "box": [1.0, 2.0, 60.0, 12.0], "confidence": 0.8,
         "low_confidence": False, "confirm_numbers": True, "script": "printed", "has_numbers": True},
        {"text": "Amoxicillin 25 mg", "box": [1.0, 20.0, 70.0, 34.0], "confidence": 0.97,
         "low_confidence": False, "confirm_numbers": True, "script": "handwritten", "has_numbers": True},  # sure of itself, still flagged
        {"text": "see overleaf", "box": [1.0, 40.0, 50.0, 52.0], "confidence": 0.31,
         "low_confidence": True, "confirm_numbers": False, "script": "handwritten", "has_numbers": False}]
    assert body["preprocessing"] == {"quarter_turns": 3, "orientation_confidence": 0.9876, "deskew_degrees": 0.0,
                                     "flip_check": {"margin": 0.31, "crops": 2, "flipped": False},
                                     "deskew_requested": False, "deskew_applied": False}
    assert body["models"]["handwriting"] == "checkpoints/handwriting_v1"
    assert body["words"][0] == {"text": "Total", "confidence": 0.9, "box": [1.0, 2.0, 30.0, 12.0], "line": 0}
    assert body["page"] == {"width": 120, "height": 90}


def test_v1_defaults_for_flags_and_preprocessing(client, monkeypatch):
    class Unsure(FakePipeline):
        def run(self, image):
            res = super().run(image)
            res["lines"][0]["confidence"] = 0.42
            return res

    monkeypatch.setattr(api, "_pipeline", Unsure())
    body = post(client, png_bytes()).get_json()
    assert body["pipeline"] == "v1"
    assert body["lines"][0]["low_confidence"] is True  # below 0.5
    assert body["lines"][0]["confirm_numbers"] is True and body["lines"][0]["script"] == "printed"
    assert body["preprocessing"] == {"quarter_turns": 0, "orientation_confidence": None, "deskew_degrees": 0.0,
                                     "flip_check": None, "deskew_requested": False, "deskew_applied": False}


def test_deskew_and_scanner_flags_are_per_request_with_v2(client, monkeypatch):
    fake = FakeScanner()
    monkeypatch.setattr(api, "_pipeline", fake)
    body = post(client, png_bytes(), deskew="1", scanner="1").get_json()
    assert fake.calls[-1]["deskew"] is True and fake.calls[-1]["use_scanner"] is True
    assert body["preprocessing"]["deskew_degrees"] == 1.25
    assert body["preprocessing"]["deskew_requested"] is True and body["preprocessing"]["deskew_applied"] is True
    # nothing is left switched on in the shared object once the request is over
    assert fake.do_deskew is False and fake.ocr.use_scanner is False
    body = post(client, png_bytes()).get_json()  # the next request, without flags
    assert fake.calls[-1]["deskew"] is False and fake.calls[-1]["use_scanner"] is False
    assert body["preprocessing"]["deskew_degrees"] == 0.0 and body["preprocessing"]["deskew_applied"] is False
    assert post(client, png_bytes(), deskew="0").status_code == 200
    assert fake.calls[-1]["deskew"] is False


def test_flags_do_not_leak_when_reading_fails(client, monkeypatch):
    class Broken(FakeScanner):
        def run(self, image):
            raise RuntimeError("out of memory")

    fake = Broken()
    monkeypatch.setattr(api, "_pipeline", fake)
    assert post(client, png_bytes(), deskew="1", scanner="1").status_code == 500
    assert fake.do_deskew is False and fake.ocr.use_scanner is False


def test_deskew_with_v1_is_accepted_and_reported_as_not_applied(client, monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(api, "_pipeline", fake)
    r = post(client, png_bytes(), deskew="1")
    assert r.status_code == 200
    pre = r.get_json()["preprocessing"]
    assert pre["deskew_requested"] is True and pre["deskew_applied"] is False and pre["deskew_degrees"] == 0.0
    assert not hasattr(fake, "do_deskew")  # nothing was put on the v1 object


@pytest.mark.parametrize("value", ["maybe", "2", "-1", "yes please"])
def test_bad_deskew_flag_is_400(client, monkeypatch, value):
    fake = FakeScanner()
    monkeypatch.setattr(api, "_pipeline", fake)
    r = post(client, png_bytes(), deskew=value)
    assert r.status_code == 400 and r.get_json()["error"] == "deskew must be 1 or 0"
    assert fake.calls == []


def test_v2_is_chosen_when_its_package_and_files_are_there(client, monkeypatch, tmp_path):
    v1_ready(monkeypatch)
    v2_ready(monkeypatch, tmp_path)
    loads = []
    monkeypatch.setattr(api, "_load_v2", lambda: loads.append("v2") or FakeScanner())
    monkeypatch.setattr(api, "_load_v1", lambda: loads.append("v1") or FakePipeline())
    status = client.get("/api/ocr/status").get_json()
    assert status["available"] is True and status["loaded"] is False and loads == []  # status loads nothing
    assert status["pipeline"] == "v2" and "pipeline_reason" not in status
    assert status["capabilities"] == {"handwriting": True, "orientation": True, "deskew": True}
    assert set(status["models"]) == {"detector", "recogniser"}
    body = post(client, png_bytes()).get_json()
    assert body["pipeline"] == "v2" and loads == ["v2"]
    assert post(client, png_bytes()).status_code == 200 and loads == ["v2"]  # loaded once
    status = client.get("/api/ocr/status").get_json()
    assert status["loaded"] is True and status["pipeline"] == "v2"
    assert status["models"] == FakePipeline.name  # the printed pipeline inside the Scanner


@pytest.mark.parametrize("packages, missing, words", [
    (("a_package_that_is_not_installed_xyz",), (), "requirements-handwriting.txt"),
    (("json",), ("handwriting",), "handwriting"),
    (("json",), ("orientation", "printed / handwritten routing"), "orientation, printed / handwritten routing"),
])
def test_v1_is_chosen_when_v2_cannot_run(client, monkeypatch, tmp_path, packages, missing, words):
    v1_ready(monkeypatch)
    v2_ready(monkeypatch, tmp_path, packages, missing)
    loads = []
    monkeypatch.setattr(api, "_load_v2", lambda: loads.append("v2") or FakeScanner())
    monkeypatch.setattr(api, "_load_v1", lambda: loads.append("v1") or FakePipeline())
    status = client.get("/api/ocr/status").get_json()
    assert status["available"] is True and status["pipeline"] == "v1"
    assert status["capabilities"] == {"handwriting": False, "orientation": False, "deskew": False}
    assert words in status["pipeline_reason"]
    body = post(client, png_bytes()).get_json()
    assert body["pipeline"] == "v1" and loads == ["v1"]  # v2 was not even tried
    assert all(ln["confirm_numbers"] is True and ln["script"] == "printed" for ln in body["lines"])
    status = client.get("/api/ocr/status").get_json()
    assert status["loaded"] is True and status["pipeline"] == "v1" and words in status["pipeline_reason"]


def test_v1_is_used_when_v2_fails_to_load(client, monkeypatch, tmp_path):
    v1_ready(monkeypatch)
    v2_ready(monkeypatch, tmp_path)
    loads = []

    def broken():
        loads.append("v2")
        raise MemoryError("not enough memory for the handwriting model")

    monkeypatch.setattr(api, "_load_v2", broken)
    monkeypatch.setattr(api, "_load_v1", lambda: loads.append("v1") or FakePipeline())
    assert client.get("/api/ocr/status").get_json()["pipeline"] == "v2"  # nothing wrong can be seen yet
    r = post(client, png_bytes())
    assert r.status_code == 200 and r.get_json()["pipeline"] == "v1"  # the request is still answered
    assert loads == ["v2", "v1"]
    status = client.get("/api/ocr/status").get_json()
    assert status["available"] is True and status["loaded"] is True and status["pipeline"] == "v1"
    assert status["capabilities"] == {"handwriting": False, "orientation": False, "deskew": False}
    assert "could not be loaded (MemoryError)" in status["pipeline_reason"]
    assert post(client, png_bytes()).status_code == 200 and loads == ["v2", "v1"]  # no second attempt


def test_both_pipelines_failing_is_503(client, monkeypatch, tmp_path):
    v1_ready(monkeypatch)
    v2_ready(monkeypatch, tmp_path)

    def broken():
        raise OSError("unreadable checkpoint")

    monkeypatch.setattr(api, "_load_v2", broken)
    monkeypatch.setattr(api, "_load_v1", broken)
    r = post(client, png_bytes())
    assert r.status_code == 503 and "could not be loaded (OSError)" in r.get_json()["error"]
    status = client.get("/api/ocr/status").get_json()
    assert status["available"] is False and status["pipeline"] == "v1"


def test_status_when_unavailable_still_reports_the_pipeline(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "missing_packages", lambda: ["torch"])
    v2_ready(monkeypatch, tmp_path, ("a_package_that_is_not_installed_xyz",))
    body = client.get("/api/ocr/status").get_json()
    assert body["available"] is False and body["reason"]
    assert body["pipeline"] == "v1" and body["capabilities"]["handwriting"] is False
    assert "requirements-handwriting.txt" in body["pipeline_reason"]


def test_capability_check_does_not_import_the_heavy_packages():
    code = ("import sys\n"
            "from ocr import api\n"
            "api.v2_problem(); api.pipeline_info(); api.availability()\n"
            "print(any(m in sys.modules for m in ('torch', 'doctr', 'transformers')))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=api.ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"


# ---------------------------------------------------------------- image modes and the smallest image

def _upload(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class Keeps(FakePipeline):
    """Remembers the image it was given."""

    def run(self, image):
        self.image = image.copy()
        return super().run(image)


def test_transparent_png_reaches_the_reader_as_dark_text_on_white(client, monkeypatch):
    fake = Keeps()
    monkeypatch.setattr(api, "_pipeline", fake)
    im = Image.new("RGBA", (120, 90), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([20, 30, 100, 40], fill=(20, 20, 20, 255))
    assert post(client, _upload(im)).status_code == 200
    assert fake.image.mode == "RGB"
    assert fake.image.getpixel((5, 5)) == (255, 255, 255)  # transparent -> paper, not black
    assert fake.image.getpixel((60, 35)) == (20, 20, 20)  # the text is still there


def test_16_bit_grey_png_is_scaled_not_clamped(client, monkeypatch):
    import numpy as np
    fake = Keeps()
    monkeypatch.setattr(api, "_pipeline", fake)
    a = np.full((90, 120), 32768, np.uint16)  # a mid-grey page
    a[30:40, 20:100] = 2570  # dark text
    assert post(client, _upload(Image.fromarray(a))).status_code == 200
    assert fake.image.mode == "RGB"
    assert fake.image.getpixel((5, 5)) == (128, 128, 128)
    assert fake.image.getpixel((60, 35)) == (10, 10, 10)


@pytest.mark.parametrize("size", [(1, 1), (2, 2), (15, 15), (15, 200), (200, 15)])
def test_tiny_images_are_400(client, monkeypatch, size):
    fake = FakePipeline()
    monkeypatch.setattr(api, "_pipeline", fake)
    r = post(client, png_bytes(size))
    assert r.status_code == 400
    assert r.get_json()["error"] == "image is too small: each side must be at least 16 pixels"
    assert fake.calls == []


def test_smallest_accepted_image(client, monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(api, "_pipeline", fake)
    assert post(client, png_bytes((16, 16))).status_code == 200
    assert fake.calls[-1]["size"] == (16, 16)

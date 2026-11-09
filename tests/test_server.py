"""Tests for server.py: static files, what is never served, JSON errors under /api/, scanner registration."""
from __future__ import annotations

import io
import sys
import types

import flask
import pytest

import server


@pytest.fixture()
def app():
    """App without the scanner blueprint, so these tests do not depend on Agent B's code."""
    a = server.create_app(with_scanner=False)
    a.config["TESTING"] = True
    return a


@pytest.fixture()
def client(app):
    return app.test_client()


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"<title>Health Diary</title>" in r.data
    assert r.mimetype == "text/html"
    assert client.get("/index.html").status_code == 200


@pytest.mark.parametrize("path, mimetype", [
    ("/css/styles.css", "text/css"),
    ("/js/app.js", "text/javascript"),
    ("/js/storage.js", "text/javascript"),
    ("/js/files-db.js", "text/javascript"),
])
def test_static_files_served(client, path, mimetype):
    r = client.get(path)
    assert r.status_code == 200
    assert r.mimetype in (mimetype, "application/javascript")


def test_index_references_only_local_assets(client):
    html = client.get("/").get_data(as_text=True)
    assert "http://" not in html.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "https://" not in html
    for src in ("js/storage.js", "js/files-db.js", "js/ui.js", "js/documents.js", "js/scan.js", "js/voice.js", "js/app.js"):
        assert src in html
        assert client.get("/" + src).status_code == 200


@pytest.mark.parametrize("path", [
    "/server.py",
    "/tests/test_server.py",
    "/README.md",
    "/requirements.txt",
    "/pytest.ini",
    "/docs/PLAN.md",
    "/models/",
    "/scanner/api.py",
    "/data/",
    "/.git/config",
    "/css/",
    "/js/",
    "/css/../server.py",
    "/js/..%2fserver.py",
    "/js/%2e%2e/server.py",
    "/css/styles.css.bak",
])
def test_private_files_and_listings_not_served(client, path):
    r = client.get(path)
    assert r.status_code == 404
    assert b"import" not in r.data  # never the source of server.py or the scanner


def test_security_headers(client):
    r = client.get("/")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    csp = r.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "http" not in csp  # no remote origins allowed


def test_api_unknown_is_json_404(client):
    r = client.get("/api/nothing-here")
    assert r.status_code == 404
    assert r.is_json and "error" in r.get_json()
    assert r.headers["Cache-Control"] == "no-store"


def test_scan_without_scanner_is_json_404(client):
    for r in (client.get("/api/scan/status"), client.post("/api/scan")):
        assert r.status_code == 404
        assert r.is_json
        assert "scanner" in r.get_json()["error"].lower()


def test_api_too_large_is_json_413(app):
    bp = flask.Blueprint("echo", __name__)

    @bp.post("/api/_test_upload")
    def upload():
        f = flask.request.files["image"]
        return {"size": len(f.read())}

    app.register_blueprint(bp)
    c = app.test_client()
    small = c.post("/api/_test_upload", data={"image": (io.BytesIO(b"x" * 1000), "a.png")},
                   content_type="multipart/form-data")
    assert small.status_code == 200 and small.get_json() == {"size": 1000}
    big = c.post("/api/_test_upload", data={"image": (io.BytesIO(b"x" * (server.MAX_REQUEST_BYTES + 1)), "a.png")},
                 content_type="multipart/form-data")
    assert big.status_code == 413
    assert big.is_json and "too large" in big.get_json()["error"]


def test_api_method_not_allowed_is_json(app):
    bp = flask.Blueprint("only_post", __name__)

    @bp.post("/api/_only_post")
    def only_post():
        return {"ok": True}

    app.register_blueprint(bp)
    r = app.test_client().get("/api/_only_post")
    assert r.status_code == 405
    assert r.is_json and "error" in r.get_json()
    assert "POST" in r.headers["Allow"]


def test_api_unexpected_error_is_json_500(app):
    bp = flask.Blueprint("boom", __name__)

    @bp.get("/api/_boom")
    def boom():
        raise RuntimeError("secret detail")

    app.register_blueprint(bp)
    app.config["TESTING"] = False  # let the error handler run instead of propagating
    r = app.test_client().get("/api/_boom")
    assert r.status_code == 500
    assert r.is_json and r.get_json() == {"error": "Internal server error."}


def test_server_starts_when_scanner_import_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "scanner.api", None)  # makes "from scanner.api import ..." raise
    a = server.create_app()
    assert a.config["SCANNER_REGISTERED"] is False
    assert a.test_client().get("/").status_code == 200
    assert a.test_client().get("/api/scan/status").status_code == 404


def test_fake_scanner_blueprint_is_registered(monkeypatch):
    bp = flask.Blueprint("scan", __name__)

    @bp.get("/api/scan/status")
    def status():
        return {"available": True, "model_version": "test", "metrics": None}

    pkg = types.ModuleType("scanner")
    pkg.__path__ = []
    mod = types.ModuleType("scanner.api")
    mod.scan_bp = bp
    monkeypatch.setitem(sys.modules, "scanner", pkg)
    monkeypatch.setitem(sys.modules, "scanner.api", mod)
    a = server.create_app()
    assert a.config["SCANNER_REGISTERED"] is True
    r = a.test_client().get("/api/scan/status")
    assert r.status_code == 200 and r.get_json()["available"] is True


@pytest.mark.parametrize("env, expected", [
    ({}, ("127.0.0.1", 8000)),
    ({"HOST": "0.0.0.0", "PORT": "8101"}, ("0.0.0.0", 8101)),
    ({"HOST": "", "PORT": ""}, ("127.0.0.1", 8000)),
])
def test_bind_address(env, expected):
    assert server.bind_address(env) == expected


@pytest.mark.parametrize("port", ["0", "70000", "abc"])
def test_bind_address_rejects_bad_port(port):
    with pytest.raises(ValueError):
        server.bind_address({"PORT": port})

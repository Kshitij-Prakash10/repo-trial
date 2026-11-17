"""Local server for Health Diary: serves the static app and the document scanner API.

Run:  python server.py            (http://127.0.0.1:8000)
Env:  HOST (default 127.0.0.1), PORT (default 8000).

It runs on waitress (a production WSGI server with connection limits and timeouts) when that is
installed, else on Flask's development server. Requests under /api/ that come from another website
(an Origin header that is not this server) are refused, and when the server listens only on this
computer, requests addressed to any other host name are refused too (DNS rebinding).

Only the app's own static files are served (index.html, css/, js/); never this file, tests, models or
data. Errors under /api/ are JSON ({"error": "..."}). The scanner blueprint (scanner/api.py) is optional:
if it cannot be imported the app still runs and /api/scan answers 404 JSON. The OCR blueprint (ocr/api.py)
is registered the same way; its heavy packages are loaded on the first POST /api/ocr, not at start-up.
Nothing here needs the internet.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Tuple
from urllib.parse import urlsplit

from flask import Flask, Response, abort, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_REQUEST_BYTES = 11 * 1024 * 1024  # bytes; the scanner accepts photos up to 10 MB
STATIC_DIRS = ("css", "js")  # the only folders served, besides index.html
STATIC_SUFFIXES = (".css", ".js", ".svg", ".png", ".ico", ".webp", ".woff2")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")  # HOST values that mean "this computer only"
SERVER_THREADS = 8  # waitress worker threads
CHANNEL_TIMEOUT = 60  # seconds an idle or half-sent connection is kept

log = logging.getLogger("healthdiary.server")

# Same-origin only: no CDN, no external fonts, no remote APIs. blob: and data: are for the user's own
# files (thumbnails, previews, audio) that live in the browser.
CSP = "; ".join([
    "default-src 'self'",
    "img-src 'self' data: blob:",
    "media-src 'self' blob:",
    "frame-src 'self' blob:",
    "object-src 'self' blob:",
    "style-src 'self'",
    "script-src 'self'",
    "connect-src 'self'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


def bind_address(env: "os._Environ[str] | dict | None" = None) -> Tuple[str, int]:
    """(host, port) from env HOST and PORT, default ("127.0.0.1", 8000). Raises ValueError on a bad port."""
    env = os.environ if env is None else env
    host = (env.get("HOST") or "127.0.0.1").strip()
    port_text = (env.get("PORT") or "8000").strip()
    port = int(port_text)
    if not 0 < port < 65536:
        raise ValueError(f"PORT must be 1..65535, got {port_text}")
    return host, port


def register_scanner(app: Flask) -> bool:
    """Register scanner.api.scan_bp if it imports. Returns True when registered."""
    try:
        from scanner.api import scan_bp  # Agent B's blueprint: POST /api/scan, GET /api/scan/status
        app.register_blueprint(scan_bp)
        return True
    except Exception as err:  # the scanner is optional; the diary must still run without it
        log.warning("Scanner not available (%s: %s). /api/scan will answer 404.", type(err).__name__, err)
        return False


def register_ocr(app: Flask) -> bool:
    """Register ocr.api.ocr_bp (POST /api/ocr, GET /api/ocr/status) if it imports. Returns True when registered."""
    try:
        from ocr.api import ocr_bp  # light import: torch and docTR are loaded on the first request
        app.register_blueprint(ocr_bp)
        return True
    except Exception as err:  # reading text is optional; the diary must still run without it
        log.warning("OCR not available (%s: %s). /api/ocr will answer 404.", type(err).__name__, err)
        return False


def _hostname(netloc: str) -> str:
    """Host name of a Host header or URL authority, lower case, without port or IPv6 brackets."""
    try:
        return (urlsplit("//" + netloc).hostname or "").lower()
    except ValueError:
        return ""


def create_app(with_scanner: bool = True, with_ocr: bool = True,
               allowed_hosts: Optional[Iterable[str]] = None) -> Flask:
    """Build the Flask app. `with_scanner` / `with_ocr` False skip those blueprints (for tests).

    `allowed_hosts`: host names (without port) requests may be addressed to; None accepts any.
    """
    hosts = None if allowed_hosts is None else {h.lower() for h in allowed_hosts}
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    app.config["SCANNER_REGISTERED"] = register_scanner(app) if with_scanner else False
    app.config["OCR_REGISTERED"] = register_ocr(app) if with_ocr else False

    @app.before_request
    def same_site_only():
        # A page on another website can make the browser send requests here. It cannot read the answers,
        # but it could keep the scanner and the text reader busy, so such requests are refused.
        if hosts is not None and _hostname(request.host) not in hosts:
            return Response("This server only answers requests addressed to this computer.",
                            status=421, mimetype="text/plain")
        origin = request.headers.get("Origin")
        if origin and request.path.startswith("/api/"):
            o = urlsplit(origin)
            if o.scheme not in ("http", "https") or o.netloc.lower() != request.host.lower():
                return jsonify({"error": "Requests from other websites are not accepted."}), 403
        return None

    @app.get("/")
    @app.get("/index.html")
    def index() -> Response:
        return send_from_directory(ROOT, "index.html")

    @app.get("/<any(css, js):folder>/<path:name>")
    def static_file(folder: str, name: str) -> Response:
        if folder not in STATIC_DIRS or not name.lower().endswith(STATIC_SUFFIXES):
            abort(404)
        # send_from_directory refuses paths that leave the folder (../) with 404.
        return send_from_directory(os.path.join(ROOT, folder), name)

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status=204)

    @app.errorhandler(HTTPException)
    def http_error(err: HTTPException):
        if request.path.startswith("/api/"):
            if err.code == 413:
                message = f"The upload is too large (at most {MAX_REQUEST_BYTES // (1024 * 1024)} MB per request)."
            elif err.code == 404 and request.path.startswith("/api/scan") and not app.config["SCANNER_REGISTERED"]:
                message = "The scanner is not installed on this server."
            elif err.code == 404 and request.path.startswith("/api/ocr") and not app.config["OCR_REGISTERED"]:
                message = "Reading text is not installed on this server."
            elif err.code == 404:
                message = "Not found."
            else:
                message = err.description or err.name
            resp = jsonify({"error": message})
            resp.status_code = err.code or 500
            if getattr(err, "valid_methods", None):
                resp.headers["Allow"] = ", ".join(err.valid_methods)
            return resp
        return err

    @app.errorhandler(Exception)
    def unexpected_error(err: Exception):
        if isinstance(err, HTTPException):
            return http_error(err)
        log.exception("Unexpected error on %s", request.path)
        if request.path.startswith("/api/"):
            return jsonify({"error": "Internal server error."}), 500
        return Response("Internal server error.", status=500, mimetype="text/plain")

    @app.after_request
    def headers(resp: Response) -> Response:
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Content-Security-Policy", CSP)
        # Health data: never let a shared cache keep API answers; the static app is revalidated each load.
        resp.headers.setdefault("Cache-Control", "no-store" if request.path.startswith("/api/") else "no-cache")
        return resp

    return app


def main() -> None:
    """Start the server on HOST:PORT."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    host, port = bind_address()
    local_only = host in LOOPBACK_HOSTS
    app = create_app(allowed_hosts=LOOPBACK_HOSTS if local_only else None)
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    ocr = "not installed"
    if app.config["OCR_REGISTERED"]:
        from ocr.api import availability
        ok, reason = availability()
        ocr = "ready" if ok else f"not available: {reason}"
    print(f"Health Diary: http://{shown}:{port}  (scanner {'ready' if app.config['SCANNER_REGISTERED'] else 'not installed'})")
    print(f"Reading text: {ocr}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("Note: the server is reachable from other devices on the network. Recording audio in the browser "
              "needs http://127.0.0.1, http://localhost or https.")
    try:
        from waitress import serve
    except ImportError:
        print("Note: waitress is not installed (pip install -r requirements.txt), so Flask's development "
              "server is used. It has no connection limits or timeouts.")
        app.run(host=host, port=port, debug=False, threaded=True)
        return
    serve(app, host=host, port=port, threads=SERVER_THREADS, channel_timeout=CHANNEL_TIMEOUT,
          max_request_body_size=MAX_REQUEST_BYTES, ident=None)


if __name__ == "__main__":
    main()

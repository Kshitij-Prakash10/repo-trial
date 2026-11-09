# Plan: health documents, voice notes and an ML document scanner

Health Diary is a browser app: every user's records live in the browser (`js/storage.js`, localStorage).
This change adds, per user, a **document library** (prescriptions, lab reports, medical certificates,
discharge summaries, insurance papers, other), **voice notes**, and a **document scanner** whose page
detector is a trained machine-learning model. Two agents build it in parallel; this file is their contract.

## Design decisions

- Documents and voice notes are stored in the browser with **IndexedDB** (localStorage is too small for
  files), keyed by the Health Diary user id. Nothing is uploaded anywhere. Deleting a user deletes their
  documents and notes.
- The scanner model is Python (numpy, scikit-learn, Pillow), so a small **Flask server** (`server.py`)
  serves the app and the scanner API on http://127.0.0.1:8000. Opening `index.html` directly still works;
  only scanning then shows "Scanner needs the local server (python server.py)".
- Offline at run time: no CDN, no web fonts, no cloud APIs.

## Ownership

| Path | Owner |
|---|---|
| `index.html`, `css/**`, `js/**`, `server.py`, `tests/test_server.py`, `tests/ui/**` | **Agent A** |
| `scanner/**`, `models/**`, `data/scanner_examples/**`, `tests/test_scanner*.py`, `docs/scanner.md` | **Agent B** |
| `README.md`, `requirements.txt`, `pytest.ini`, `.gitignore`, `docs/PLAN.md` | Coordinator |

Edit only your own paths. Put requests for the coordinator in your final report.

## Interface: the scanner

### Python (Agent B)

```python
# scanner/infer.py
def scan(image: "PIL.Image.Image", mode: str = "gray") -> dict:
    """Find the page, correct perspective, enhance. mode: "color" | "gray" | "bw".
    Returns {"page_found": bool, "corners": [[x, y] * 4] in source pixels, order TL, TR, BR, BL (image
    orientation), "confidence": float 0..1, "image": PIL.Image (the output page), "model_version": str,
    "method": "ml" | "fallback"}. Never raises on a valid image: with no model file or no page found it
    returns the whole image enhanced, page_found False, method "fallback"."""
```

### HTTP (Agent B writes the blueprint, Agent A registers it in server.py)

`scanner/api.py` defines `scan_bp = flask.Blueprint("scan", __name__)`:

- `POST /api/scan`: multipart/form-data, field `image` (JPEG, PNG or WebP, at most 10 MB), optional field
  `mode` (`color` | `gray` | `bw`, default `gray`).
  - 200: `{"page_found", "corners", "corners_norm" (0..1), "confidence", "method", "width", "height",
    "image_png_b64" (no data: prefix), "model_version", "elapsed_ms"}`
  - 400 `{"error"}`: missing field, not an image, bad mode. 413 `{"error"}`: over 10 MB.
- `GET /api/scan/status`: `{"available": bool, "model_version": str | null, "metrics": dict | null}`

`server.py` registers it so the server still starts if the scanner is missing:

```python
try:
    from scanner.api import scan_bp
    app.register_blueprint(scan_bp)
except Exception:
    pass
```

## Agent A: features and UI

1. **Documents** (new sidebar section): add a file (PDF, PNG, JPEG, WebP, at most 20 MB, type checked from
   the first bytes), title, category, document date, tags, notes, optional link to an existing visit or
   condition of that user. List with search and category filter, thumbnail for images, view, download,
   edit, delete. Show the latest documents on the Dashboard.
2. **Voice notes** (new sidebar section): record with MediaRecorder (timer, stop, preview, save or discard),
   title, optional link to a document; list, play, rename, delete. Clear messages when the microphone is
   missing, permission is denied, or the page is not a secure context; offer an audio-file upload instead.
3. **Scan a paper document** (in Documents): take a photo (`accept="image/*" capture="environment"`) or
   choose one, POST to `/api/scan`, show the detected outline (corners_norm) over the photo and the
   corrected page, switch color / gray / black-and-white, save as a PNG document. Use
   `/api/scan/status` to show when the scanner is not available.
4. **Storage**: a new `js/files-db.js` (IndexedDB) used only through a small API, like `js/storage.js`.
   Show storage used per user. Deleting a user removes their files.
5. **Server**: `server.py` (Flask) serves the static app and registers the scanner blueprint; HOST and PORT
   env vars, default 127.0.0.1:8000; JSON errors for /api/*; `MAX_CONTENT_LENGTH` 11 MB.
6. **UI improvement** across the whole app: keep the existing look and structure but polish it: consistent
   spacing and type scale, clearer forms and tables, empty and error states, responsive down to 360 px,
   keyboard and screen-reader accessible, dark theme via `prefers-color-scheme`, no external assets.
   Every existing feature keeps working with existing users' data (do not change the localStorage shape
   except by adding fields).
7. **Tests**: `tests/test_server.py` (pytest) and a Playwright check script under `tests/ui/` that drives
   the main flows in headless Chromium (create user, add vitals, add a document, save a voice note from an
   uploaded file, scan flow against the real or a mocked `/api/scan`).

## Agent B: the ML document scanner

1. **Data**: seeded synthetic generator (`scanner/synth.py`): health documents (prescriptions, lab
   reports, certificates, discharge summaries, insurance forms) photographed on varied backgrounds with
   perspective, rotation, scale, partial crops, lighting, shadows, blur, noise, JPEG artefacts. Labelled
   synthetic everywhere.
2. **Model**: trained page segmenter (numpy, scipy, scikit-learn, Pillow only), committed under
   `models/` (under 10 MB), under 1 s per photo on CPU; quad fit, perspective warp, enhancement.
   Compare against a non-ML baseline on the same held-out set.
3. **Evaluation**: held-out by seed; mask IoU, corner error as % of the page diagonal, success rate
   (IoU >= 0.9), model vs baseline, in `models/scanner_metrics.json`. No invented numbers.
4. **Commands**: `python -m scanner.train` (deterministic, under 5 minutes on a laptop CPU).
5. **Integration**: `scan()` and `scanner/api.py` exactly as above, with tests.
6. **Docs**: `docs/scanner.md`.

A later step fine-tunes the scanner on real photos (DocCornerDataset) and tests it on SmartDoc 2015
Challenge 1; keep `scanner/train.py` reusable for that (functions for features, training and evaluation
that take arrays of images and corners, not only the synthetic generator).

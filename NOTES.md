# OCR scanner: working notes

Branch `claude/ocr-scanner`. These notes record what was found in the repository and on the machine
before anything was changed, and the decisions that follow from it.

## Step 1: what exists

### The current scanner (`scanner/`)

It is **not a neural network** and uses no deep-learning framework. It is a page finder built on
scikit-learn, numpy, scipy and Pillow.

| Part | File | What it does |
|---|---|---|
| Features | `scanner/features.py` | Splits the photo into a grid, 128 cells on the long side, and computes 39 numbers per cell (colour, brightness, edges, texture, position). |
| Model | `models/scanner_real.joblib` (app default), `models/scanner.joblib` (synthetic only) | Two `HistGradientBoostingClassifier` stages. Stage 1: P(cell is page) from the 39 features. Stage 2: the same plus smoothed stage-1 output from neighbouring cells. |
| Geometry | `scanner/geometry.py` | Cleans the probability map, fits a 4-corner quad, refines the corners on image edges, warps the quad to a flat rectangle (homography). |
| Enhancement | `scanner/enhance.py` | Shading removal and contrast stretch; modes `color`, `gray`, `bw`. |
| Inference | `scanner/infer.py` | `scan(image, mode)` and `detect(image, model)`. |
| HTTP | `scanner/api.py`, `server.py` | `POST /api/scan` (optional manual `corners`), `GET /api/scan/status`. |
| Training | `scanner/train.py` | Trains the synthetic model from `data/synthetic/` (generator `scanner/synth.py`, seeds fixed). |
| Fine-tuning | `scanner/finetune/` | Adds trees on real photos (DocCornerDataset); report in `docs/finetune/results.md`. |

**Input:** a PIL image (any mode; converted to RGB; photos over 2,400 px on the long side are reduced).

**Output of `scan()`:** a dict with
- `page_found` (bool), `corners` (TL, TR, BR, BL in source pixels), `confidence` (0 to 1),
- `image` (PIL, the flattened and enhanced page),
- `model_version`, `method` (`"ml"` or `"fallback"`).

When no page is found (confidence below 0.35, quad under 5 % of the photo, or not convex) it returns the
whole photo, enhanced, with `page_found: false`. It does not raise on a valid image.

**It produces no text.** There is no text detection or recognition anywhere in the repository.

**Measured behaviour that matters for OCR** (from `docs/finetune/results.md`, held-out data):
- On real photos the page quad is good on average but not exact: SmartDoc mean Jaccard 0.766, and only
  50 % of frames at or above 0.9. It fails on a light page on a light surface.
- A wrong quad would crop text away. The OCR pipeline must therefore not trust the scanner blindly (see
  the design below).
- Flatbed scans and born-digital pages have no visible page border. The scanner is trained on photos of
  a page on a surface, so on such inputs it may report no page or a wrong quad.

**Config and checkpoints:** constants in the Python modules (no config files); checkpoints are the
joblib files above plus `models/finetune/*.joblib` (ablations) and `models/onnx/` (export). None of them
is touched by the OCR work.

**Tests:** `python -m pytest -q` (120 tests) and `node tests/ui/check.js` (25 UI checks).

### Compute and versions (measured on this machine)

| Item | Value |
|---|---|
| CPU | 4 cores |
| RAM | 15 GB |
| GPU | none (`nvidia-smi` not present) |
| Free disk | 29 GB of 252 GB |
| OS / Python | Ubuntu 24.04, Python 3.13.18 |
| Existing stack | numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, Pillow 12.3.0, Flask 3.1.3, joblib 1.6.0 |
| Deep-learning framework | none installed before this work |
| Tesseract | not installed before this work; 5.3.4 available from apt |
| Network | PyPI, apt, Hugging Face, `doctr-static.mindee.com`. GitHub release downloads and the ICDAR site are not reachable. |

**What this means.** Everything must train and run on 4 CPU cores. Free disk (29 GB) is already under
the 30 GB ceiling given for data, so the data and tools together must stay far below it.

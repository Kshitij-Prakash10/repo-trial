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

## Step 2: pipeline design

```
input image
  -> text detection on the uncropped photo        (needed to check the scanner's crop)
  -> page scanner (existing model, unchanged)     -> crop and flatten, only when safe
  -> text detection on the flattened page         (only when the crop was used)
  -> text recognition on each word box
  -> reading order (lines top to bottom, words left to right)
  -> plain text + JSON (words and lines with boxes and confidences)
```

**The scanner stage is reused as it is** (`scanner.infer.detect` and `scanner.geometry.warp`, same
weights, no retraining). Its crop is used only when all of these hold (`ocr/pipeline.py`):
- a page is found with confidence of at least 0.5;
- the quad covers at most 90 % of the photo (otherwise there is nothing to gain);
- at least 98 % of the words the detector finds on the uncropped photo lie inside the quad.

Otherwise the whole photo goes on to OCR. The reason is the scanner's measured accuracy (Step 1): a
wrong quad would silently cut text away, and scans or screenshots have no page border to find. These
three thresholds were fixed before any test-set run and were not tuned on the test sets.

**Framework: docTR 1.1.0 on PyTorch 2.15 (CPU).** Reasons:
- It has pretrained English document detection and recognition models that are small enough to
  fine-tune and run on 4 CPU cores.
- PaddleOCR's weights are hosted on a site this machine cannot reach.
- TrOCR-small is a 62 M-parameter autoregressive transformer: it reads one word at a time and is too
  slow here for pages with hundreds of words.

**Models, chosen on validation pages only** (pretrained weights, no fine-tuning; scratch measurements
on 20 pages / 946 word crops from the FUNSD and CORD validation splits):

| Detector | Params | s/page (CPU) | FUNSD val F1 | CORD val F1 |
|---|---|---|---|---|
| db_mobilenet_v3_large | 4.2 M | 0.69 | 0.851 | 0.715 |
| fast_tiny | 8.5 M | 0.48 | 0.851 | 0.851 |
| **fast_base (chosen)** | 10.6 M | 0.60 | **0.865** | **0.903** |
| linknet_resnet18 | 11.5 M | 0.34 | 0.851 | 0.897 |
| db_resnet50 | 25.4 M | 1.23 | 0.847 | 0.905 |

| Recogniser | Params | crops/s (CPU) | val CER | val WER |
|---|---|---|---|---|
| crnn_mobilenet_v3_small | 2.1 M | 790 | 0.048 | 0.149 |
| **crnn_mobilenet_v3_large (chosen)** | 4.5 M | 526 | 0.050 | 0.148 |
| crnn_vgg16_bn | 15.8 M | 96 | 0.046 | 0.137 |
| parseq | 23.8 M | 40 | 0.047 | 0.133 |
| vitstr_small | 21.4 M | 50 | 0.051 | 0.144 |

- **Detector:** `fast_base` has the best validation F1 at 0.6 s per page.
- **Recogniser:** all five are within about half a point of CER before fine-tuning, and the two
  MobileNet CRNNs are 5 to 20 times faster than the others. `crnn_mobilenet_v3_large` was taken over
  the small one for its extra capacity during fine-tuning; a full pass over the training crops costs
  minutes rather than hours.

**Scope:** English printed text. The recogniser's vocabulary (126 characters) also covers French
accents, which comes for free with the pretrained weights. Handwriting is out of scope.

## Step 3: data

| Dataset | Source (Hugging Face, pinned revision) | Licence | Pages train / val / test | Annotation |
|---|---|---|---|---|
| FUNSD | `nielsr/funsd` | FUNSD licence: non-commercial research and education | 129 / 20 / 50 | words |
| SROIE (ICDAR 2019) | `jsdnrs/ICDAR2019-SROIE` | CC BY 4.0 | 566 / 60 / 361 | text lines |
| CORD v2 | `naver-clova-ix/cord-v2` | CC BY 4.0 | 800 / 100 / 100 | words |

- **Test splits** are the official ones and are never used for training or for any choice. CORD has an
  official validation split. FUNSD and SROIE have none, so 20 and 60 pages of their official train
  splits are set aside as validation (seed 2026).
- **Size:** about 2.9 GB downloaded and 2.8 GB extracted, far below the 30 GB limit.
- **FUNSD boxes** in this copy are on a 0 to 1000 scale, not pixels (found by drawing them); the
  converter scales them back.
- **SROIE is annotated per line.** For word-level training and scoring its lines are split into words
  by a fixed rule (`ocr/data.py:split_line`): the widest runs of empty columns in the binarised line are
  the spaces. 99.6 % of lines split (136 of 53,012 could not be and are left out of the word-level
  sets). The split boxes are approximate, so SROIE's word-level detection and recognition numbers carry
  some label noise; its end-to-end numbers use the original line transcriptions.
- **CORD** receipts are Indonesian (Latin script, mostly numbers and menu items), not English.

**Skipped:**
- **ICDAR 2013 / 2015 scene text.** The official site needs a registered account and is not reachable
  from this machine. A third-party copy of the ICDAR 2015 word crops exists on Hugging Face, but it is
  street-scene text, not documents, so it was left out.
- **SROIE from the official site** (registration); the Hugging Face copy above was used instead.
- **Synthetic text lines.** The brief asks for them if real data is small. The three datasets give
  about 100,000 real word crops for recognition, so none were generated.

# Document scanner

The scanner turns a phone photo of a paper health document into a flat, cleaned-up page. It finds the
page with a trained machine-learning model, corrects the perspective, and enhances the result in colour,
gray or black-and-white. It runs locally on the CPU (numpy, scipy, scikit-learn, Pillow); nothing is sent
anywhere.

There are two models:
- **`models/scanner.joblib`** is trained and tested only on synthetic photos (rendered documents on
  rendered backgrounds). This document describes it, and the numbers below are its held-out results on
  synthetic data.
- **`models/scanner_real.joblib`** is that model fine-tuned on real phone photos (DocCornerDataset) and
  tested on SmartDoc 2015 Ch1. **The app uses it when it is present.** Its results, including where it is
  worse than the synthetic model, are in `docs/finetune/results.md`.

## Use

```python
from PIL import Image
from scanner.infer import scan
res = scan(Image.open("photo.jpg"), mode="gray")   # "color" | "gray" | "bw"
res["image"].save("page.png")    # res: page_found, corners (TL TR BR BL, px), confidence, method, model_version
```

HTTP (blueprint `scanner.api.scan_bp`, registered by `server.py`): `POST /api/scan` (multipart field
`image`, optional `mode`) and `GET /api/scan/status`, as specified in `docs/PLAN.md`. The API accepts
JPEG, PNG and WebP up to 10 MB, applies the EXIF orientation, rejects images over 50 megapixels, and
works on a copy reduced to at most 2400 px on the long side; corners are returned in the pixels of the
uploaded (EXIF-rotated) image.

If neither model file is present, or no plausible page is found, `scan()` returns the whole photo
enhanced with `page_found: false` and `method: "fallback"`. It does not raise on a valid image.

## Data (synthetic)

`scanner/synth.py` renders personal health papers (prescription, lab report with a results table,
medical certificate, discharge summary, insurance claim form with boxed character cells) with headers,
real text from Pillow's built-in font, text-like line blocks, tables, checkboxes, stamps and signatures,
on white, off-white or tinted paper with slight curl shading. The page is warped onto a background (wood,
fabric, solid, a desk close to the paper colour, gradients, blotches, stripes, tiles, dots, plus clutter
such as phones, books and other papers) with random rotation, keystone perspective and corner jitter, the
page covering about 20 to 90 percent of the photo, sometimes cut off by the frame, sometimes with a thumb
or pen over an edge. Then: soft drop shadow, lighting gradient, sometimes a hand or phone shadow, colour
cast, blur, sensor noise and JPEG compression (quality 35 to 95). Photos are 480 to 1280 px on the long
side. Ground truth: the 4 page corners; the mask is their polygon.

The exact sets used are committed in `data/synthetic/` (see its README): 360 train, 60 val, 200 test
photos, plus 80 `ood` photos from a generator setting never used in training (other layouts: referral
letter, pharmacy bill, blood-sugar log; chequered tablecloth and granite backgrounds; stronger tilt;
warm light). Splits are by generator seed and disjoint.

## Model

1. **Features** (`scanner/features.py`): the photo is box-downscaled to a grid with long side 128 cells.
   Per cell, 39 features: RGB, brightness, saturation, chroma, opponent colours, chromaticity; colour
   distance from the median of the border ring and of the centre (in Lab-like and in chromaticity space);
   brightness rank and brightness relative to the local maximum (15 and 31 cells); local means at 5, 11
   and 25 cells and local standard deviations at 3, 7 and 15 cells; gradient magnitude; fine texture
   (gradient on a 4x finer grid, averaged per cell; text makes pages textured); normalised position and
   distance from the border.
2. **Classifier**: `HistGradientBoostingClassifier` (200 iterations, 31 leaves) predicts page or
   background per cell (stage 1). An auto-context stage 2 (same classifier type) gets the features plus
   smoothed stage-1 probabilities at several scales; it is trained on out-of-fold stage-1 maps (2 folds by
   photo) so that it learns from realistic stage-1 errors. 3000 cells are sampled per training photo.
3. **Quadrilateral fit** (`scanner/geometry.py`): threshold 0.5, opening, largest connected component,
   hole fill; sub-cell boundary points interpolated from the probability map; the convex hull simplified
   to 10 vertices gives candidate lines (each refitted on nearby boundary points by total least squares);
   every 4 of them form a candidate quad, scored by soft IoU with the probability map + 0.5 x the fraction
   of boundary points on its sides - a penalty for angles further than 25 degrees from 90. Lines extend
   past corners cut off by the frame.
4. **Edge refinement**: on a 640 px copy of the photo, each side snaps to the strongest colour edge within
   1.5 grid cells (robust line fit, rejected if it turns more than 4 degrees).
5. **Warp**: `Image.transform(PERSPECTIVE)`; output size from the side lengths, snapped to the A4 ratio
   when within 15 percent; at most 2000 px.
6. **Enhancement** (`scanner/enhance.py`): shading removal (divide by a max-filtered, blurred background
   estimate), contrast stretch; `bw` is an adaptive threshold on a lightly denoised page.
7. **Confidence**: mean page probability inside the quad x (1 - mean outside) x an angle plausibility
   term. `page_found` needs confidence >= 0.35 and a quad covering at least 5 percent of the photo.

**Baseline** (non-ML): Otsu threshold on grid brightness, then exactly the same component, quad fit and
edge refinement. The comparison therefore isolates what the learned page mask adds.

Model file `models/scanner.joblib`: 704 KB.

## Held-out results (copied from `models/scanner_metrics.json`)

Mask IoU: IoU of the predicted quad (clipped to the frame) with the true page polygon at full resolution.
Corner error: mean distance of the 4 corners to the true ones, in percent of the true page diagonal (best
cyclic correspondence, so a page lying sideways is not penalised for which corner is called top-left).
Success: IoU >= 0.9. A fallback counts with the image corners as its prediction.

**Test set, 200 synthetic photos (seeds 200000 to 200199):**

| | Model | Baseline (Otsu) |
|---|---|---|
| Mask IoU, mean / median | 0.968 / 0.993 | 0.814 / 0.936 |
| Corner error % of diagonal, mean / median / p90 | 1.63 / 0.21 / 4.12 | 10.39 / 3.06 / 35.69 |
| Success rate (IoU >= 0.9) | **0.945** | 0.570 |
| Page found | 1.000 | 0.890 |

By background (test, success rate model / baseline): light desk close to the paper colour 0.90 / 0.00
(n = 21), tiles 0.82 / 0.12 (17), stripes 0.94 / 0.35 (17), blotch 0.93 / 0.75 (28), dots 0.96 / 0.72
(25), fabric 0.95 / 0.68 (22), gradient 0.96 / 0.68 (28), solid 1.00 / 0.68 (25), wood 1.00 / 0.94 (17).
Pages cut off by the frame (n = 42): 0.90 / 0.55, median corner error 2.14 / 4.83 percent (the cut-off
corners are extrapolated, so they are less precise).

Ablations on the test set: without the auto-context stage, success 0.935 and median corner error 0.78
percent; without edge refinement, 0.935 and 0.66 percent.

**Out-of-distribution set, 80 synthetic photos from the `ood` generator setting (never trained on):**

| | Model | Baseline (Otsu) |
|---|---|---|
| Mask IoU, mean / median | 0.847 / 0.933 | 0.671 / 0.732 |
| Corner error % of diagonal, mean / median / p90 | 7.00 / 3.35 / 15.15 | 16.37 / 8.49 / 46.95 |
| Success rate (IoU >= 0.9) | **0.588** | 0.288 |

By background (ood, success model / baseline): chequered tablecloth 0.39 / 0.15 (n = 33), granite 0.91 /
0.78 (23), light desk 0.54 / 0.00 (24). The drop from 0.945 to 0.588 shows that the model does not
generalise fully to backgrounds and layouts it has not seen; a light-and-dark chequered cloth, whose light
squares look like paper, is the worst case.

Validation set (60 photos): model 0.900 success, median corner error 0.13 percent; baseline 0.567.

Speed: the full `scan()` (detect, warp, gray enhancement) took 297 ms mean, 464 ms p90, 541 ms max per
photo on 25 test photos (4-core x86_64 cloud machine shared with another heavy job at the time). A 20
megapixel JPEG through the HTTP API took about 2.5 s, mostly decoding and PNG encoding.

Training (`python -m scanner.train`): 341 s in total on that shared machine, of which 97 s generated
`data/synthetic/` (skipped when it exists), 10 s features, 151 s fitting (an earlier development fit on 300 photos took 48 s on the same
machine when it was idle), the rest evaluation and examples. Deterministic: fixed seeds everywhere, and
two fits on the same rows give identical probabilities.

Examples (`data/scanner_examples/`, photo with the true outline in green and the predicted one in red,
next to the gray output): `test_200001.png`, `test_200004.png`, `test_200007.png`, `test_200010.png`
(a failure on light tiles: the outline is wrong and the page is distorted), `ood_300001.png`,
`ood_300002.png`.

## Known limits

- Trained and tested on synthetic photos only. Real photos have paper texture, folds, real lighting,
  motion blur and backgrounds the generator does not produce. Expect lower accuracy until it is
  evaluated (and fine-tuned) on real photos; `scanner/train.py` has `image_rows`, `build_rows`, `fit`,
  `evaluate` and `report` that take any list of `{"image", "corners", "meta"}` items for that.
- Backgrounds with paper-like light patches (chequered cloth, white tiles, other white papers touching the
  page) are the main failure mode.
- Pages are assumed flat: folded or curled pages are rectified as a plane.
- No text-orientation detection: a page photographed sideways comes out sideways.
- The confidence score is weakly informative: on the test set every photo was above the 0.35 threshold;
  on the ood set, photos above it succeeded 59 percent of the time and the one below it failed.
- Thin, faint strokes can break up in `bw` mode.

## Retrain

```bash
python -m scanner.dataset --force   # optional: regenerate data/synthetic/ (byte-for-byte)
python -m scanner.train             # trains from data/synthetic/, writes models/scanner.joblib,
                                    # models/scanner_metrics.json and data/scanner_examples/
```

## AI and open-source use

Written with Claude Code (Anthropic, model Claude Opus 5.5). Open-source packages: numpy, scipy
(ndimage, ConvexHull), scikit-learn (HistGradientBoostingClassifier), joblib, threadpoolctl (installed
with scikit-learn), Pillow (rendering, its built-in font, warping, JPEG/PNG), Flask, pytest. No external
datasets and no pretrained models are used.

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
   term. `page_found` needs confidence >= 0.60 and a quad covering at least 5 percent of the photo.
   The 0.60 was chosen on the 1,000 DocCornerDataset validation photos (fine-tuned model) as the highest
   value that keeps at least 99.5 percent of the well-outlined pages (IoU >= 0.9). Until then it was
   0.35, which every number in this document and in `docs/finetune/results.md` was measured with. From
   the stored per-photo results, 0.35 -> 0.60 changes: false alarms on photos without a document,
   validation 37 % -> 4 %, test 31 % -> 7 %; pages reported found, validation 99 % -> 94 %, test
   97 % -> 79 %. On test, about two thirds of the pages no longer reported had a poor outline
   (IoU < 0.5); 99 % of the well-outlined ones are kept. A page that is not reported found is shown
   whole, and the app offers to set the corners by hand.

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
- The confidence score is weakly informative on synthetic photos (measured at the earlier 0.35
  threshold): on the test set every photo was above the 0.35 threshold;
  on the ood set, photos above it succeeded 59 percent of the time and the one below it failed.
- Thin, faint strokes can break up in `bw` mode.

## Guards and image modes (scanner/infer.py, scanner/convert.py)

- **Image modes.** `scan` converts its input with `scanner.convert.to_rgb` (also used by the OCR
  pipeline as `ocr.pipeline.to_rgb`): fully transparent areas become white, partly transparent ones
  keep their own colour (`alpha_to_rgb`; an image with no opaque pixel at all is put on white), 16-bit
  grey (`I;16`) is scaled from
  0..65535 to 0..255, float and 32-bit grey are scaled by their largest value, and negative floats are
  shifted so the lowest value is black. A plain `convert("RGB")` made transparent images black and
  16-bit scans white. `enhance(..., "color")` no longer turns a blank (flat) image black.
  Compositing partly transparent pixels on white (as first done) lightened a photo's half see-through
  surroundings and hid the page edge: on 20 synthetic test photos with alpha 128 outside the page,
  IoU >= 0.9 0.40 and mean IoU 0.856, against 0.75 and 0.907 when their colour is kept (the same as a
  plain `convert("RGB")`). Compositing them on the median colour of the opaque border instead gave
  0.80 / 0.921 there but lost a page (found 1.00 -> 0.90) when the see-through surroundings were black,
  so the simpler rule was kept. Fully transparent cut-outs and 16-bit grey are unchanged (0.95 / 0.953
  and 0.50 / 0.751).
- **Tiny images.** A photo whose shorter side is under 32 px (`MIN_SIDE`) is never a page.
- **Noise guard.** Uniform random RGB noise used to reach confidence ~0.63 and `page_found=True`. A quad
  that passes the confidence test is now rejected when the photo inside and outside it is
  indistinguishable on all three counts (`coherence`, `looks_like_noise`): mean colour d' < 0.05, mean
  gradient (texture) ratio within 10 %, and no side with a coherent colour step (step z < 4). On 220
  noise photos (uniform and Gaussian, 5 sizes, 22 seeds) the found ones had d' <= 0.023, texture ratio
  within 1.5 % and step z <= 2.7; on all 700 stored synthetic photos (train, val, test, ood) with a quad,
  d' >= 0.068. Check on the stored synthetic test (200) and ood (80) photos, before and after the guard,
  measured with `models/scanner_real.joblib`: identical photo for photo (test: 189/200 found, mean IoU
  0.887; ood: 46/80 found, mean IoU 0.673). Its effect on real photos without a document (DocCorner
  negatives) has not been measured, because those photos are not in the repository; the guard can only
  turn "found" into "not found".
- **Blank guard.** A plain paper-coloured photo with a lighting gradient passed the noise guard: the
  gradient is a small but coherent step along the quad's sides, and inside and outside differ in mean
  colour. `looks_blank` rejects a quad when the strongest side's mean colour step is below 0.03
  (`BLANK_MAX_STEP`, RGB 0..1) and the inside texture is below 0.03 (`BLANK_MAX_TEXTURE`, mean gradient
  at 320 px). Found quads on paper blanks had a step <= 0.010 and texture <= 0.013; all 625 synthetic
  photos with a quad (train, val, test, ood) have a step >= 0.013 and texture >= 0.046. Paper-coloured
  blanks with a lighting gradient reported as a page: 6/12 -> 0/12 (and 6/12 -> 0/12 on a held-out
  seed); synthetic desk backgrounds without a page: 4/50 -> 3/50 (the three left have a real colour
  step); white, grey and dark blanks, noise and tiny images: 0 before and after. Synthetic test and
  ood: identical photo for photo. The effect on DocCorner and SmartDoc cannot be measured here (the
  photos are not in the repository), and the stored per-photo results cannot show it either, because
  the guard needs the photo.

## Recall at MIN_CONFIDENCE 0.60, and corner bloat: what the stored results allow

- **A lower threshold is not recommended yet.** From the stored DocCorner validation results (880
  pages, 120 photos without a document; the guards above cannot be applied to them), pages found /
  false alarms: 0.50: 97.8 % / 22.5 %; 0.55: 96.8 % / 12.5 %; 0.60: 94.4 % / 4.2 %. The best lower
  candidate, 0.55, triples the false alarms on validation, so 0.60 stays. Test, looked at once for
  0.55: found 87.2 % (78.7 % at 0.60), false alarms 11.3 % (6.7 %); SmartDoc found 95.9 % (90.5 %). At
  0.60, 99.1 % of the test pages with IoU >= 0.9 are still found: most pages no longer reported had a
  poor outline. Revisit with the guards' features stored per real photo (plan below).

`python reports/scanner_recall/study.py` (output in `reports/scanner_recall/study.txt`) uses only the
committed per-photo results of the shipped model; the photos are not in the repository.

- **Quad geometry does not recover the lost recall.** Accepting confidence 0.35 to 0.60 when the
  predicted quad looks like a page (angles within 10 degrees of 90, opposite sides within 20 %, no
  corner outside the photo), chosen on validation, gives validation found 0.944 -> 0.951 with false
  alarms 0.042 -> 0.050; test (read once) found 0.786 -> 0.794, false alarms 0.067 -> 0.075. That is
  one point of recall bought with one point of false alarms, so it is not used.
- **Corner bloat on SmartDoc background01.** True quads are not stored. Against the synthetic-only
  baseline's quad on frames where the baseline is very accurate (iou_doc >= 0.95, 382 frames), the
  shipped model's quad is more than 10 % larger in 24.6 % of them (their iou_doc 0.824 vs 0.935 overall),
  and their confidence is still high (median 0.83), so no confidence threshold separates them. Nothing
  was changed.
- **Plan (needs the DocCorner val/test and SmartDoc photos, `scanner/finetune/download.py`):** for every
  photo, store per-side edge support (`geometry._support` on the grid boundary points, and the
  refined-edge gradient strength from `refine_edges`), the noise guard's d' / texture ratio / step z,
  and interior fine-texture density; fit a small logistic model of "IoU >= 0.9" vs "negative or IoU <
  0.5" on validation only; pick the operating point that keeps validation false alarms <= 4.2 % and
  report test once. For bloat, add a per-side check that the refined side lies on a step between page
  colour and background colour (step z per side, already computed by `coherence`) and pull a side
  inward to the strongest such step when the outer one has low support; validate on SmartDoc
  background01/04 and DocCorner val before changing any default.

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

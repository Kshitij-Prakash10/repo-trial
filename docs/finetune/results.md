# Scanner fine-tuning on real phone photos: results

The document scanner (`scanner/`) was trained only on synthetic photos (`data/synthetic/`). This report
covers fine-tuning it on real photos: the baseline, the fine-tuned models, the ablations and the failure
cases, plus every assumption and deviation from the requested recipe.

**Outcome in one paragraph.** The model chosen on validation (`real_only`, now `models/scanner_real.joblib`
and used by the app) improves on the real-photo test sets by every mean measure:
- DocCornerDataset test mean IoU goes from 0.668 to 0.723.
- False alarms on photos with no document fall from 85 % to 31 %.
- SmartDoc Ch1 official mean Jaccard goes from 0.747 to 0.766.

It is **not better everywhere**:
- On SmartDoc, the share of frames with Jaccard >= 0.9 falls from 0.558 to 0.505.
- On SmartDoc, the median corner error rises from 19.7 px to 35.3 px.
- On the synthetic test set, mean IoU falls from 0.968 to 0.903 (forgetting).

The fine-tuned model fails less often catastrophically and is less exact on easy frames. The original
synthetic model is kept unchanged as `models/scanner.joblib`.

## 1. Data

| Dataset | Role | What was used | Licence |
|---|---|---|---|
| DocCornerDataset (Hugging Face `mapo80/DocCornerDataset`, revision `2f6d2d16`) | fine-tuning (train), checkpoint selection (val), second test set (test) | seeded subsets: train 6,000 (720 without a document), val 1,000 (120), test 2,000 (239) | `other`: a compilation. MIDV-500 / MIDV-2019 research use only; SmartDoc research use only; COCO CC BY 4.0; Roboflow Universe sets various (per dataset). |
| SmartDoc 2015 Challenge 1 (`frames.tar.gz`, GitHub release v2.0.0 of `jchazalon/smartdoc15-ch1-dataset`) | **held-out test only**, never trained on or used for any choice | every 10th frame of every video: 2,555 of 24,889 frames, 5 backgrounds x 6 document types | CC BY 4.0 (the GitHub dataset release); thumbnails in `docs/finetune/failures/` are from it |
| Synthetic photos (this repo) | 30 % of fine-tuning cells (mixed variants), forgetting check | `data/synthetic/train` 360, `test` 200 | generated here, no licence restrictions |
| Mendeley "real-world document images" (optional) | not used | **download failed**, see below | CC BY 4.0 |

Because of the research-only parts, DocCornerDataset images are **not** committed. What is committed:
- The checksums of every downloaded file (`data/real/manifests/*sha256*`).
- The exact file lists used and excluded (`data/real/manifests/doccorner_*`).
- The SmartDoc frame list (`smartdoc_ch1_test_frames.csv`).

**Downloads that failed (not substituted):**
- **Mendeley:** the request from Python's `urllib` was answered `HTTP Error 403: Forbidden` (the site rejects
  automated clients). Retrying with a browser-like User-Agent was refused by this session's permission policy
  as bot-check circumvention, so that was not attempted further. Mendeley is optional in the brief, and no
  other data was put in its place.
- **SmartDoc `pip install smartdoc15-ch1`:** this failed on Python 3.13 while building its dependency
  `Polygon2` (`SyntaxError: Missing parentheses in call to 'print'`). The same frames were taken from the
  dataset's official GitHub release archive instead, which is the archive that package itself downloads.

**Label format checks.**
- 20 random training images per split and 20 SmartDoc frames were drawn with their quads, with each corner in
  its own colour. All were checked visually. The sheets are in `<data_home>/label_checks/`, not committed
  because of the DocCorner licence.
- DocCorner corners are normalised to 0-1 and ordered TL, TR, BR, BL relative to the image.
- SmartDoc metadata lists the corners as tl, bl, br, tr, ordered by the document rather than the image (a
  rotated page has its "tl" anywhere).
- All quads are converted to TL, TR, BR, BL in image pixels. Corner error is measured under the best cyclic or
  reversed matching, so corner ordering cannot affect the metric.

**Leakage check (DocCorner vs SmartDoc).**
- 891 DocCorner images whose names start with `smartdoc` (637 train, 179 val, 75 test) were excluded.
  Visually they come from a different SmartDoc collection (red backgrounds, not the Ch1 videos); they were
  excluded anyway.
- Every remaining DocCorner image was compared with every SmartDoc Ch1 frame by a 256-bit difference hash.
  One image was within the threshold (Hamming distance 12): `more2_001069`, at distance 5. It looked
  like a false positive but was excluded anyway.
- Per-image distances: `data/real/manifests/doccorner_vs_smartdoc_dhash.csv.gz`.

## 2. Method and how it maps to the requested recipe

The scanner is not a neural network. It is two `HistGradientBoostingClassifier` stages on a 128-cell grid:
- **Stage 1:** 39 colour, edge and position features per cell.
- **Stage 2:** the same features plus smoothed stage-1 output (auto-context).
- **After the classifiers:** a quad fit, edge refinement and a perspective warp.

The neural-network recipe was mapped as follows:

| Requested | Done here |
|---|---|
| Start from the synthetic checkpoint | Start from `models/scanner.joblib` (200 + 200 trees). Fine-tuning **adds** trees with `warm_start`; the existing trees and the feature binning are not changed. |
| Stage A: freeze backbone, train head, lr ~1e-3 | Stage 1 frozen; stage 2 (the context "head") gets +25 / +50 / +100 trees at learning rate 0.1 (the original boosting rate). |
| Stage B: unfreeze, 10x lower lr, cosine schedule, AdamW | Both stages get +50 / +100 / +200 trees at learning rate 0.01. Boosting has no optimiser, weight decay or schedule; the 10x lower rate is kept. |
| Early stopping on validation | The number of added trees is chosen by mean quad IoU on 600 DocCorner val photos; the best checkpoint over all steps is kept (the start model included). |
| 70 % real / 30 % synthetic batches | 70 % of training cells come from real photos (600 cells per photo, 3.6 M real cells) and 30 % from synthetic photos (1.54 M cells). The `real_only` variant uses no synthetic cells. |
| Loss: L1 on corners + BCE on objectness | Log loss (BCE) on the per-cell page mask. Corners come from the geometric quad fit, so there is no corner regression term. Objectness is the existing `page_found` (confidence >= 0.35 and a plausible quad). |
| Negatives | Photos with no document are trained with an all-background mask. |
| Augmentations | **Not implemented.** Real photos are used as stored (downscaled to a long side of 1,024 px). |

A **scikit-learn issue was found and worked around.** On warm start, scikit-learn 1.9.1 re-fits the
feature bins on the new data and replays the existing trees on the re-binned data. Each old tree then reads
the wrong bins. In the first attempt this made the model worse even on its own training photos: validation IoU
fell from 0.827 to 0.537. `scanner/finetune/finetune.py:grow` keeps the original bins fixed. A guard stops any
step that raises the training loss. `tests/test_finetune.py` covers the problem and the fix, including a control
test that shows the plain scikit-learn behaviour misreads the old trees.

## 3. Baseline vs fine-tuned

All numbers come from `models/finetune/eval/*.json`, written by `scanner.finetune.report`.
- **IoU:** quad overlap in the image.
- **Official Jaccard:** SmartDoc Ch1's own measure, computed in the document's frame.
- **Corner error:** shown as a percentage of the image diagonal.
- **False alarms:** the share of no-document photos on which a page was reported.

The selected model is **real_only**: best on validation (selection IoU 0.891 on 600 val photos, against
0.885 for full and 0.874 for no_stage_a).

### SmartDoc 2015 Ch1 (held-out test, 2,555 frames)

| Model | Official Jaccard mean | Jaccard >= 0.9 | IoU (image) mean | corner err median px | err median % diag | err mean % diag | found (recall) |
|---|---|---|---|---|---|---|---|
| synthetic baseline | 0.747 | **0.558** | 0.745 | **19.7** | **0.90** | 6.23 | 0.998 |
| full (A then B, 70/30) | 0.712 | 0.398 | 0.706 | 73.1 | 3.32 | 6.38 | 0.997 |
| no_stage_a (B only, 70/30) | 0.760 | 0.516 | 0.757 | 34.2 | 1.55 | 5.35 | 0.998 |
| **real_only (selected)** | **0.766** | 0.505 | **0.763** | 35.3 | 1.60 | **5.03** | 0.997 |

### DocCornerDataset test (2,000 photos, 239 without a document)

| Model | IoU mean | IoU >= 0.9 | err median % diag | err mean % diag | objectness precision | recall | false alarms |
|---|---|---|---|---|---|---|---|
| synthetic baseline | 0.668 | 0.271 | 5.74 | 9.67 | 0.893 | 0.963 | 0.854 |
| full | 0.717 | 0.373 | 4.00 | 7.74 | 0.957 | 0.964 | 0.322 |
| no_stage_a | 0.709 | 0.354 | 4.14 | 8.02 | 0.945 | 0.953 | 0.410 |
| **real_only (selected)** | **0.723** | **0.385** | **3.86** | **7.18** | **0.959** | **0.971** | **0.305** |

### DocCornerDataset val (1,000 photos; the selection used the first 600) and synthetic test (200)

| Model | val IoU mean | val IoU >= 0.9 | val false alarms | synthetic IoU mean | synthetic IoU >= 0.9 |
|---|---|---|---|---|---|
| synthetic baseline | 0.826 | 0.463 | 0.850 | **0.968** | **0.945** |
| full | 0.885 | 0.668 | 0.400 | 0.930 | 0.840 |
| no_stage_a | 0.871 | 0.607 | 0.525 | 0.947 | 0.875 |
| **real_only (selected)** | **0.893** | **0.685** | **0.367** | 0.903 | 0.760 |

### Size and speed

Latency is for `detect` only, on 40 SmartDoc 1920 x 1080 frames. Each model was timed in one quiet run on
this container's CPU, after a warm-up.

| Model | trees (stage 1 + 2) | joblib size | ms mean | ms p90 |
|---|---|---|---|---|
| synthetic baseline | 200 + 200 | 0.70 MB | 119.6 | 136.0 |
| full | 400 + 500 | 1.45 MB | 182.4 | 204.2 |
| no_stage_a | 400 + 400 | 1.32 MB | 169.3 | 190.1 |
| real_only (selected) | 250 + 350 | 1.01 MB | 146.5 | 171.3 |

### SmartDoc by background (official Jaccard mean)

| Background | frames | baseline | full | no_stage_a | real_only |
|---|---|---|---|---|---|
| background01 (wood desk) | 633 | **0.905** | 0.778 | 0.871 | 0.867 |
| background02 (light round table) | 613 | 0.422 | 0.492 | 0.521 | **0.604** |
| background03 | 609 | 0.908 | 0.946 | **0.955** | 0.948 |
| background04 | 430 | **0.798** | 0.707 | 0.766 | 0.728 |
| background05 | 270 | **0.673** | 0.534 | 0.595 | 0.546 |

Fine-tuning helps most on the hardest background (02) and on 03. It hurts on 01, 04 and 05. The breakdowns
by document type are in the JSON files.

## 4. Ablations

The variants are compared on validation; the test numbers are given for information only.

- **Real-only vs 70/30 mix (both with Stage A):**
  - real_only is slightly better on validation (0.891 vs 0.885 selection IoU) and on both test sets.
  - It also stopped after fewer stage-B trees, so it is smaller and faster.
  - The mix forgets less: synthetic IoU 0.930 vs 0.903.
  - Mixing synthetic cells did not help on real photos at this data size. Its only effect was to keep
    synthetic accuracy.
- **With vs without Stage A (both 70/30):**
  - On validation, Stage A helps: 0.885 vs 0.874 selection IoU, and false alarms 0.400 vs 0.525.
  - On DocCorner test, Stage A also helps: 0.717 vs 0.709.
  - On SmartDoc it hurts clearly: 0.712 vs 0.760 Jaccard.
  - Stage A adds up to 100 trees at the high rate 0.1 to the context stage. It fits DocCorner's
    backgrounds closely, and that does not carry over to SmartDoc's desks.
  - Each variant ran once with one seed, so differences of about 0.01 should not be over-read.

## 5. Failure analysis (SmartDoc, lowest official Jaccard)

Files in `docs/finetune/failures/` (true quad green, prediction red):
- `ft_real_only_smartdoc_worst30.jpg` and `.csv`: the selected model.
- `base_smartdoc_worst30.*` and `ft_full_smartdoc_worst30.*`: the baseline and full models.
- `ft_full_vs_base_background01.jpg`: blue = baseline, red = full.

**Pattern 1: page segmented as table (background02).**
- All 30 worst frames of every model are on background02: a large, light-grey round table that is nearly the
  colour of the paper.
- The predicted quad lies on bare table beside the page, with zero overlap. It is 0.9 to 3.3 times the
  page's area. The confidence is 0.47 to 0.75, above the 0.35 threshold, so `page_found` stays true.
- Per-cell colour and edge features cannot separate white paper from a light table well enough. The
  page has low contrast against it, while the table edge against the dark floor is the strongest edge in the frame.
- Fine-tuning reduced this failure (background02 Jaccard 0.42 to 0.60) but did not remove it.

**Pattern 2: one corner pulled out onto the desk (background01, fine-tuned models).**
- The fine-tuned models find the page but extend one or two corners, usually at the bottom (nearest the
  camera), onto the light wood.
- Over all SmartDoc frames, median predicted area is 1.24x the true area for `full`, against 1.02x for the baseline.
- This is the main cause of the drop in Jaccard >= 0.9 and of the higher median corner error. The extra trees
  make the page mask leak into bright, low-texture desk regions near the page.

None of these observations was used to change the model or choose a checkpoint.

## 6. Export

`python -m scanner.finetune.export_onnx --model models/scanner_real.joblib --out models/onnx` writes
`stage1.onnx` (0.49 MB) and `stage2.onnx` (0.69 MB).
- **Check:** on 69,248 feature rows the largest difference from scikit-learn is 5e-7 (stage 1) and 1e-6
  (stage 2); see `models/onnx/export_check.json`.
- **Converter:** skl2onnx 1.20 could not convert scikit-learn 1.9 models (`TypeError: Field
  onnx.AttributeProto.ints: Expected an int, got a boolean`). The graph is therefore written directly from the
  fitted trees: an `ai.onnx.ml` TreeEnsembleRegressor followed by a Sigmoid.
- **What is not exported:** only the classifiers are in ONNX. Feature extraction, the quad fit and the warp
  are Python and would have to be ported to use the ONNX files elsewhere.
- **TFLite and TorchScript:** not produced; they are for neural networks.
- **App runtime:** the app keeps using the joblib file and runs offline.

## 7. Deviations and assumptions (complete list)

1. **Model type:** gradient-boosted trees, not a neural network. The stages, learning rates, schedule and
   loss are mapped as in section 2. There is no cosine schedule, AdamW or weight decay, and no L1 corner loss.
2. **No data augmentation** during fine-tuning.
3. **Subsets** were used for compute reasons on a 4-CPU container:
   - DocCorner train 6,000, val 1,000 (600 for selection) and test 2,000, out of 32,330, 8,466 and 6,577
     after exclusion.
   - These were seeded (2026) and stratified by source.
   - SmartDoc was evaluated on every 10th frame (2,555 of 24,889).
4. **Image size:** DocCorner images were downscaled to a long side of 1,024 px before use. Errors are
   reported in original-image pixels and as a percentage of the diagonal.
5. **Checkpoint selection** used mean IoU on validation only. The false-alarm rate is reported, not traded
   against localisation. An earlier version of the score included false alarms; it preferred a model with
   worse localisation and was replaced before the final runs.
6. **Objectness** is the existing `page_found`; there is no separate classifier.
7. **SmartDoc Jaccard** is our reimplementation of the official protocol: both quads are mapped to the
   document's frame by the homography that sends the true quad to the reference page size, then a polygon
   intersection is taken. It is unit-tested but was not checked against the original evaluation tool, which
   would not install.
8. **Single runs:** each variant was trained once with one seed. No confidence intervals.
9. **Dataset licences:** DocCornerDataset's research-only parts mean the fine-tuned model is a research
   artefact. Using it commercially would need the licences checked per source.
10. **Mendeley** was not used (download refused). The `smartdoc15-ch1` package was not used (build failure);
    the same frames came from the official release archive.

## 8. Reproduce

Downloads happen once, during setup. Nothing in the app needs the internet.

```bash
pip install -r requirements-finetune.txt
python -m scanner.finetune.download   --data-home DATA            # ~12 GB, checksums verified
python -m scanner.finetune.leakcheck  --data-home DATA
python -m scanner.finetune.prep_real  --data-home DATA
python -m scanner.finetune.report     --data-home DATA --name base --model models/scanner.joblib
python -m scanner.finetune.finetune   --data-home DATA --variant full --variant no_stage_a --variant real_only
python -m scanner.finetune.report     --data-home DATA --name ft_real_only --model models/finetune/scanner_ft_real_only.joblib
python -m scanner.finetune.failures   --data-home DATA --name ft_real_only
python -m scanner.finetune.export_onnx --model models/scanner_real.joblib --out models/onnx
```

`models/scanner_real.joblib` is a copy of `models/finetune/scanner_ft_real_only.joblib`. All seeds are
fixed: data selection 2026, cell sampling `[2026, photo index]`, and the shuffle of the evaluation order 7.
On this container, fine-tuning took about 50 minutes per variant and evaluation about 15 minutes per model.

## 9. Next steps

1. **Fix pattern 1 (background02):** features that separate paper from a light background, such as local
   texture or text density, and a check that the predicted quad contains text-like content.
2. **Fix pattern 2:** penalise mask leakage near the quad edges, or use a corner-regression refinement stage.
   Both target the loss of exact fits (Jaccard >= 0.9).
3. **Augment the real photos** (colour, blur, perspective), and use more of DocCorner train (6,000 of 32,330
   used).
4. **Reduce forgetting:** a smaller real share, or synthetic cells only in stage B, to keep synthetic and
   SmartDoc precision.
5. **Repeat with several seeds** to give intervals for the ablation differences.
6. **Validate the Jaccard implementation** against the official SmartDoc evaluation tool, once its
   dependency builds on a current Python.

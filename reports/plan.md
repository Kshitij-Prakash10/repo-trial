# Plan: handwriting and rotated pages

## What the audit found (Step 0)

- **Current system** (`ocr/pipeline.py`): page scanner (gradient-boosted trees, unchanged) ->
  docTR `fast_base` word detector (pretrained) -> docTR `crnn_mobilenet_v3_large` word recogniser
  (CRNN + CTC, 4.5 M parameters, fine-tuned on printed words) -> lines by grouping word boxes.
- It reads **words**, 32 characters at most, printed only. It has no orientation or skew step.
- **Baseline** on the printed test sets is in `reports/baseline.json` (end-to-end CER 0.098 FUNSD,
  0.102 SROIE, 0.118 CORD; 2 to 3 s per page).
- **Compute:** 4 CPU cores, 15 GB RAM, **no GPU**, about 13 GB of free disk.
- The printed checkpoint is frozen in `checkpoints/printed_v1_frozen/`.

## Choice for Step 2: Option A, sized for a CPU

A separate handwriting recogniser, with a small classifier that routes each text line to the
printed or the handwriting reader.

- **Handwriting reader:** `microsoft/trocr-small-handwritten` (62 M parameters), fine-tuned.
  Measured before any training: IAM test lines CER 0.063 (120 lines), 0.18 s per line on this CPU, so
  the IAM target is within reach without a GPU. On photographed handwriting (GNHK validation) the
  same model is poor (CER above 0.6 even after cleaning the crops), so fine-tuning is needed there.
  `trocr-base` was tried: more robust on GNHK without training, but 1.3 s per line and too large to
  fine-tune on a CPU. Not used.
- **Why not Option B** (retrain the existing CRNN on mixed data): the CRNN reads single words of at
  most 32 characters, handwriting data here is annotated per line, and retraining it would put the
  printed accuracy at risk. With Option A the printed reader is not touched at all, so printed
  accuracy can only change through routing mistakes, which are measured.
- **Forgetting:** the printed model stays frozen. The handwriting model still sees printed lines in
  training (5 to 10 % of each batch) so that a misrouted printed line is not read as garbage, and IAM
  lines stay in every phase so that IAM accuracy is kept while adapting to GNHK.
- **Curriculum** (`configs/ocr/hw.json`): synthetic-heavy -> IAM-heavy -> GNHK-heavy, 3,600 steps of
  8 lines, about 3 hours on this CPU. Early stopping on the mean of IAM and GNHK validation CER.
- **Detection on handwriting:** keep the pretrained word detector, group words into lines, and cut
  each line as a rotated rectangle so slanted baselines are straightened. Measured on GNHK pages
  before deciding whether it needs more.

## Step 3: orientation and skew

- Orientation classifier (0 / 90 / 180 / 270): a small CNN on the whole page, trained on rotated
  copies of the printed and handwriting training pages; compared with Tesseract OSD.
- Deskew for small angles: projection-profile search within +/- 15 degrees.
- Order: orientation -> deskew -> (page scanner) -> detection -> routing -> recognition.

## Limits set for this work

The brief left the thresholds as placeholders. Used here: stop and ask before any download over
20 GB (largest so far: 1 GB) and before any single training run longer than 6 hours. There is no
GPU, so "GPU-hours" are CPU wall-clock hours.

## Not planned

CVL and Imgur5K (see `data/DATASETS.md`). Detector fine-tuning (its overfit test failed earlier, see
`NOTES.md`); it is revisited only if handwriting pages show the detector is the bottleneck.

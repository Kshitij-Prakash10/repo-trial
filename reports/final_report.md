# Handwriting and rotated pages: final report

All numbers are from held-out test splits, scored once per system by
`python -m ocr.evaluate_v2 --task all` (summaries in `reports/eval_v2/`, per-item rows in
`runs/eval_v2/`). Every threshold was chosen on validation data. The printed test set was scored twice:
the first full run failed the printed gate (kept in `reports/eval_v2/failed_gate/printed_v1.json`).
The fixes after it were chosen on validation only (see "What went wrong first"). Hardware: 4 CPU
cores, no GPU.

## What changed

```
before: image -> page scanner -> detection -> printed recogniser -> text
after:  image -> orientation (0/90/180/270) -> 180-degree check -> [deskew, off by default]
              -> page scanner -> detection -> per-line printed / handwritten routing
              -> printed recogniser (unchanged) and, for handwritten lines, the handwriting recogniser
```

| Part | What | Where |
|---|---|---|
| Printed path | unchanged and frozen (detector pretrained, recogniser fine-tuned earlier) | `checkpoints/printed_v1_frozen/` |
| Handwriting recogniser | `microsoft/trocr-small-handwritten` fine-tuned 3,300 steps on a curriculum: synthetic lines (24 OFL fonts), then IAM, then GNHK, with printed lines replayed throughout against forgetting | `checkpoints/handwriting_v1/` |
| Routing | 0.1 M-parameter CNN on word crops gives P(handwritten); lines with mean P >= 0.5 are also read by TrOCR, whose reading wins only if it is more confident than the printed one | `checkpoints/scriptclf_v1/` |
| Orientation | docTR page-orientation MobileNetV3 fine-tuned on rotated FUNSD / SROIE / CORD / GNHK pages and synthetic IAM pages | `checkpoints/orientation_v1/` |
| 180-degree check | no model: the printed recogniser reads the 48 largest word crops as-is and turned 180 degrees; the page is turned if the turned crops read clearly better (margin 0.10) | `ocr/orientation.py` |
| Deskew | projection-profile search within +/- 15 degrees, off by default (see limits) | `ocr/orientation.py` |
| Line confidence | per line: mean word confidence (printed) or geometric mean token probability (TrOCR); lines below 0.5 get `low_confidence: true` | `ocr/scan.py` |
| Lexicon correction | optional (`Scanner(postcorrect=True)`), SymSpell edit distance 1 on lower-case words of 4+ letters | `ocr/postcorrect.py` |
| Entry point | `scan(image_path) -> (text, confidences)` | `ocr/scan.py` |

Approach: Option A (a separate handwriting model plus a router) rather than one mixed model. It left
the printed recogniser untouched, which made "no worse on printed" a matter of routing, and
TrOCR-small was the largest handwriting model that fits a CPU budget (0.19 s per line; TrOCR-base
took 1.3 s). Reasons and probes are in `reports/plan.md`; data and licences in `data/DATASETS.md`.

## Before / after

| Measure | Target | Before | After | Result |
|---|---|---|---|---|
| FUNSD page CER (printed + handwritten forms) | <= baseline + 0.5 pt | 0.0981 | 0.0986 (+0.05 pt) | pass |
| SROIE page CER | <= baseline + 0.5 pt | 0.1017 | 0.1015 (-0.02 pt) | pass |
| CORD page CER | <= baseline + 0.5 pt | 0.1183 | 0.1183 (0.00 pt) | pass |
| IAM test line CER (2,915 lines, writer-independent) | <= 8 % (stretch 5 %) | 0.0559 (pretrained TrOCR-small) | 0.0547 | pass (stretch not met) |
| IAM test line WER | report | 0.1312 | 0.1475 | worse, see limits |
| GNHK test line CER / WER (2,130 lines) | report | 0.290 / 0.673 (pretrained TrOCR-small) | 0.128 / 0.370 | |
| GNHK test page CER / WER, end to end (172 pages) | report | 0.421 / 0.861 (printed path only) | 0.259 / 0.536 | |
| Orientation accuracy, all test pages x 4 turns (2,732) | >= 99 % | 0.956 (pretrained docTR); Tesseract OSD 0.548 on 400 | 0.9927 | pass |
| Rotated pages: worst of 90/180/270 minus upright, page CER | within 1 pt | (not handled) | FUNSD 0.00, SROIE 0.00, CORD 0.00, GNHK 0.77 pt | pass |
| Time per page (printed sets) | compare | FUNSD 3.13 s, SROIE 2.32 s, CORD 2.07 s | 4.75 s, 3.41 s, 2.80 s | 1.35 to 1.52 x |
| Time per page (GNHK) | compare | 2.34 s (printed path only) | 4.30 s | 1.84 x |
| Model files | compare | 84 MB (recogniser 18 MB + detector 66 MB) | 210 MB (+ TrOCR 119 MB fp16, orientation 6 MB, router 0.3 MB) | |

Orientation by dataset (classifier alone / with the 180-degree check): FUNSD 1.000 / 1.000, SROIE
0.995 / 0.999, CORD 0.995 / 1.000, GNHK 0.951 / 0.974. Tesseract OSD failed to answer on 78 of 400
tries (counted as wrong). The orientation task simulates the scanner's two orientation steps rather
than running the scanner: no page-scanner stage, and the upside-down margin is taken as minus the
upright one. The rotated-page task, which runs the real `Scanner`, confirms it on 25 pages per set.

Lexicon post-correction, reported with and without as asked: it does not help. IAM CER 0.0547 ->
0.0550, GNHK lines 0.1281 -> 0.1286, GNHK pages 0.2586 -> 0.2590. It stays off by default.

Small skew (synthetic 3 to 10 degrees, 25 pages per set, `reports/eval_v2/rotated.json`), page CER
without / with deskew: FUNSD 0.407 / 0.123, SROIE 0.352 / 0.103, CORD 0.360 / 0.144, GNHK 0.553 / 0.338.

Training cost (4 CPU cores): handwriting about 3.4 h, orientation 34 min, router 10 min. The full
evaluation takes about 2.5 h.

## What went wrong first, and the fixes

1. **Printed gate failed on the first full run** (FUNSD +1.47, CORD +1.74 points). An ablation on
   validation pages showed two causes.
   - **Deskew** raised CER on all four validation sets, at every estimated angle (192 validation
     pages; for example CORD 0.075 -> 0.096). Changing the border fill did not help. It is now off by
     default and offered as `Scanner(deskew=True)` for pages known to be skewed, where it helps a lot
     (above).
   - **Routing by P(handwritten) alone** sent printed lines to TrOCR. The rule "keep the reading the
     models are more confident in" was the best of 19 rules tried on validation: printed sets at or
     below printed-only CER, and GNHK pages better than routing alone (0.287 -> 0.275).
2. **EXIF rotation in GNHK photos.** About a fifth of the photos carry an EXIF rotation tag that the
   annotations assume. The first models were trained on sideways crops; all three were retrained after
   the fix (`hwdata.load_image`).
   All the validation studies are in `reports/val_studies/` with their scripts in
   `scripts/val_studies/`; `python scripts/val_studies/analyse.py` reprints every table.
3. **Orientation stuck at 97.7 % on validation.** Every remaining error was a 180-degree flip. The
   recogniser-confidence check fixed most of them (validation 0.977 -> 0.997, test 0.984 -> 0.993).

## Ten worst failures

Images are in `reports/failures/`; `scripts/failure_gallery.py` rebuilds them from the evaluation rows.

| # | What | Image | Reference | Output | CER |
|---|---|---|---|---|---|
| 1 | GNHK line | ![](failures/gnhk_eng_EU_261_7.jpg) | info@willowireland.com | WOOD PUELAUMOIIM POYUI | 1.00 |
| 2 | GNHK line | ![](failures/gnhk_eng_EU_285_7.jpg) | 730 545 Lesson Phone 6 Sprint Call 2 | 100m/m/m.m.mt2m | 0.94 |
| 3 | GNHK line | ![](failures/gnhk_eng_NA_136_3.jpg) | (2years) + (3years) | (slcc. If it really ) | 0.90 |
| 4 | GNHK line | ![](failures/gnhk_eng_EU_174_12.jpg) | emergency contact information and any updates to med... | 1 1 1 2 3 1 3 0 0 0 ... | 0.87 |
| 5 | IAM line | ![](failures/iam_test_02658.jpg) | a row of dashes | sentence Valuable | 1.00 |
| 6 | IAM line | ![](failures/iam_test_02742.jpg) | he shrugged diffidently , " I like the work . One gets plent... | HE SUDUCESED OFFICENTU, " I live THE NODOL ... | 0.72 |
| 7 | IAM line | ![](failures/iam_test_01481.jpg) | became great sighs of ecstacy . | become a bad rials at economic | 0.55 |
| 8 | GNHK page | ![](failures/page_eng_AS_072.jpg) | Summary: markets . buyers, sellers of goods, services ... | p66590 CunSsoncal ... | 1.28 |
| 9 | GNHK page | ![](failures/page_eng_NA_137.jpg) | 3 How to do it Legal, (2 companies=) ... | How to do be. Nel - ) Muaxt ... | 1.18 |
| 10 | Orientation | ![](failures/orient_gnhk_eng_EU_315_1.jpg) | turned 90 degrees CCW | classifier said 270 (confidence 0.99) | - |

Patterns:
- **Lines the line crop cuts badly** (1, 2, 4). GNHK lines are cut as the rotated rectangle around
  the annotated words. Where words on one "line" are far apart or the line curves, the crop takes in
  other text, and TrOCR reads that, or loops on digits.
- **Block capitals and unusual styles** (6). The model switches to upper case for a whole line.
- **Non-text and rare tokens** (5, 3). Rows of dashes and formulas are absent from training text.
- **Whole pages** (8, 9). Page CER above 1 means the output is longer than the reference: the
  detector splits sloped handwriting into fragments, lines are grouped wrongly, and each fragment is
  read separately.
- **Orientation** (10). The 20 remaining test misses are 18 GNHK pages and 2 SROIE pages. Eight are
  GNHK pages left upside down that the 180-degree check did not catch. The other twelve are left
  sideways, by a wrong 90-degree turn or no turn. Most of the sideways pages are handwritten notes
  with little text structure. The 180-degree check cannot fix a sideways page.

## Remaining limits

- **Handwriting pages end to end are still weak** (GNHK page CER 0.26). The handwriting
  recogniser is good on clean line crops (0.128), but on whole pages the line finding is the
  bottleneck: the detector was trained on printed text, and lines are grouped from word boxes. A
  handwriting-aware line segmenter (or fine-tuning the detector on GNHK) is the next step. The
  detector's own training script still fails its overfit test (see `RESULTS.md`).
- **IAM WER got worse** (0.131 -> 0.148) while CER improved slightly. The pretrained model was itself
  trained on IAM; the curriculum traded some IAM word accuracy for a large GNHK gain. The 5 % stretch
  target is not met.
- **Deskew is off by default.** It fixes synthetic skew but hurts real photos and scans at every angle
  in validation. The cause has not been found; candidates are angle errors on photos with
  perspective, and interpolation blur on low-resolution scans. Until then, turn it on only for pages
  known to be skewed.
- **Orientation on handwriting** is 97.4 % (GNHK), below the overall 99.3 %.
- **Speed**: pages are 1.35 to 1.84 times slower (orientation, the 180-degree check, and TrOCR for
  handwritten lines). TrOCR needs about 240 MB of RAM in fp32.
- **Data limits**: no CVL or Imgur5K (unreachable or unclear licences). The IAM mirror has no writer
  ids, so writer independence relies on the official RWTH split. GNHK validation is a 10 % split of
  the official training pages. IAM is for research use only (see `data/DATASETS.md`).
- **GNHK test duplicate**: the test split holds one image twice with different annotations
  (`eng_AS_059` and `eng_AS_077`), so one of the 172 pages is scored against a reference that does
  not match it. It is kept, so the numbers stay comparable with other uses of the official split.
- **The app** (`/api/ocr`, Read text) uses `ocr.scan` when the handwriting packages and checkpoints
  are installed and the printed-only pipeline otherwise. The Handwritten / Check labels on lines and
  the "page was rotated" line are shown right after a read and are not stored with the document, and
  the picture of the page shown in the app is not turned, only the text is read upright.
- **Settings**: CPU only; the brief's 6 GPU-hour limit was applied as 6 hours of CPU wall-clock per
  run (the longest run was 3.4 h).

## Independent test (Agent C)

A separate agent re-ran the test suite and reproduced a sample of the results (IAM 300 lines, 15
printed pages per set and 48 rotated-page rows, all identical). It found no train/test overlap in
GNHK or IAM and tried 25 kinds of edge-case input, with no crash. It found these bugs, now fixed:
- `scan()` needed a docTR download on a fresh install. The orientation model now loads without
  docTR's own weights when our checkpoint exists; checked with the network blocked and only the
  README-prefetched docTR files.
- `--limit` changed which rotated pages were drawn. The full plan is now drawn first, so the published
  numbers are unchanged.
- The orientation task stored a wrong diagnostic for pages left sideways (accuracy was unaffected).
- The pretrained TrOCR revision is now pinned.
- 16-bit grey images are scaled correctly.
- Two weak tests were fixed, and an end-to-end handwriting test was added.

## AI and open-source disclosure

Code, training runs and this report were produced with Claude Code (an AI coding assistant).
Open-source components: python-doctr (Apache-2.0), Hugging Face transformers (Apache-2.0),
TrOCR-small weights (MIT), PyTorch (BSD), OpenCV (Apache-2.0), symspellpy (MIT), Tesseract
(Apache-2.0, comparison only), Google Fonts handwriting fonts (OFL). Datasets and their licences:
`data/DATASETS.md`.

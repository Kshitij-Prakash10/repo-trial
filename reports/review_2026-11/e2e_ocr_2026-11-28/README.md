# End-to-end OCR review, 2026-11-28

**Before:** `main` at `91cfd152b256f0fe0ee42589d0d561f27490f682` (PR #15). No commits followed it.
**After:** the review branch head named in `annotated_branch.json` (`git.commit`).

## Contents

All results here are **synthetic**: invented text rendered with fonts that the handwriting model saw in
training, so they are optimistic. **No real pages were scored.**

## What changed

| Area | Change | Effect on transcription |
|---|---|---|
| Field scoring (`benchmarks/fields.py`) | One occurrence matcher (`match_fields`) for `run_handwriting` and `run_full_pages`. Each predicted occurrence matches at most once. Reports exact matches, substitutions, omissions, extras, precision, recall and order-sensitive matches. Fields on inserted lines count as extras. The old metric is kept as `fields_legacy` / `legacy`. | None (evaluator only) |
| Decimals (`benchmarks/fields.py`) | A leading decimal comma is kept (",5" was read as "5"). `is_decimal` / `has_decimal` replace the dot-only checks in every benchmark path. | None (evaluator only) |
| Selection parity (`ocr/scan.py`, `ocr/pipeline.py`) | `choose_reading`, `route_share`, `is_routed` and `line_confidence` are shared by `Scanner.run` and the annotated benchmark. | `Scanner.run` unchanged; the benchmark now matches it on near-ties |
| Upload / direct parity (`tests/test_ocr_api.py`) | Same bytes through `POST /api/ocr` and through `Scanner.run`, with matched flags: same pixels, text, order, numbers, warnings and preprocessing. | None: no divergence found |
| Review fixes (after `ac3ac30`) | `route_share` averages in the classifier's float32, as `Scanner.run` did before this branch (a float64 mean put a line at exactly 0.5 just below it). `diagnose_annotated.py` compares numbers on every line, so a number invented on a line without one is an error. `case_sensitive_unit` pairs repeated fields with case-preserved units. | Routing: back to the earlier decision at the 0.5 boundary; no saved line is that close (nearest 0.4989 / 0.5011). Diagnosis and scoring: evaluator only, no published number changed |
| `run_full_pages.py` | `--page-scanner on/off` and `--deskew on/off`, recorded with the equivalent `/api/ocr` request. A zero-byte cached detector no longer passes the prerequisite check. | None |

## Defects reproduced on 91cfd15

Before the fix:
- "Metformin 500 mg and Metformin 500 mg" vs "Metformin 500 mg" scored 2/2 exact with no omission.
- Extra predicted fields and fields on inserted lines were not counted.
- With the same name twice, changing one value gave 0/2 exact.
- ",5" parsed as "5".
- A decimal-comma reference was not a decimal for `wrong_decimal` or `exact_decimal_value`.
- With identical readers, the benchmark chose "Lorazepam 0.5 mg" while `Scanner.run` chose "Lorazepam .5 mg", because the benchmark used the unrounded printed confidence.

The new tests fail on 91cfd15 and pass on the branch.

## Evaluator change on identical saved predictions

These results are in `rescore_saved_scoring_*.json`: seven committed result sources and three readings, 21 combinations, rescored without running any model.

- The legacy metric reproduces 91cfd15's field scores exactly in all 21 combinations.
- The new exact counts equal the old ones in all 21, because these synthetic lines hold one name each and no decimal comma.
- The fix adds precision, omissions, substitutions and extras.

Example: regression split, shipped reading:

| Metric | Value |
|---|---|
| Exact (old and new) | 88/104 |
| Precision | 88/93 |
| Substitutions | 5 |
| Omissions | 11 |
| Extras | 0 |

## Annotated-crop benchmark, 91cfd15 vs branch

Command: `run_handwriting.py --threads 2`, with `DOCTR_CACHE_DIR` set to an empty folder. Results are in `annotated_*.json`.

**What changed between the two runs:**
- Reader outputs, routing and chosen text are identical on all 280 lines.
- Only the stored printed confidence differs: it now follows the pipeline's formula, a 4th-decimal change on 12 validation and 15 regression lines.

| Shipped reading, unchanged by the review | Validation (113 lines) | Regression (167 lines) |
|---|---|---|
| Line CER | 0.0225 | 0.0177 |
| Page CER | 0.0217 | 0.0170 |
| Exact lines | 86/113 | 121/167 |
| Exact numeric lines | 71/96 | 93/133 |
| Field exact (recall) | 63/76 | 88/104 |
| Field precision (new) | 63/66 | 88/93 |
| Wrong decimals at confidence ≥ 0.5 / ≥ 0.9 | 1/21 / 1/10 | 4/27 / 2/12 |

**Latency, CPU, 2 threads, per line (mean / median):**
- handwriting reader: 334 / 327 ms on main, 338 / 334 ms on the branch;
- printed reader: about 39 ms;
- routing: about 4.4 ms.

Page-level p50/p95 latency needs the full-page run, which is blocked.

### Failure ranking

From `annotated_diagnosis.json`, regression split, 167 lines:

| Category | Lines |
|---|---|
| Exact | 121 |
| Both readers wrong (recognition) | 36 |
| A reader was right but not shown (selection) | 9 |
| Handwriting right but not routed | 1 |

**Numbers:**
- 21 of the 133 lines whose reference holds a number show a wrong number (`number_wrong_where_reference_has_number`).
- 0 of the 34 lines whose reference holds no number show an invented one (`number_invented_where_reference_has_none`).
- So 21 of all 167 lines show a wrong number (`number_wrong`). All 21 get the "Check the numbers" warning, so there is no silent numeric error.
- Of the 27 decimal lines, 4 are wrong, all of them warned.
- The warning also appears on 75 of the 121 lines read exactly. This is by design: every handwritten line with a number is flagged.

**Validation split, 113 lines:**
- categories: exact 86, both readers wrong 23, selection 4, routing 0;
- numbers: 12 of 96 lines with a reference number wrong, 0 of 17 lines without one with an invented number; all 12 warned.

**Conclusion.** With correct crops, recognition errors dominate. Beam search, alternative preprocessing and separator-based selection were all tried before and did not help on fresh pages. No model, threshold or decoder change is supported by this evidence, so none was made.

## Deployed configuration

- **What the app sends:** `js/ocr.js` sends only `image`, plus `deskew=1` when "Straighten tilted text" is ticked.
- **So `/api/ocr` reads with:** the page-scanner stage **off** (`read_page(use_scanner=False)`), and deskew off unless ticked.
- **What the benchmark measured:** `Scanner()` and the published `run_full_pages` results used the stage **on**. `--page-scanner off` now benchmarks the app's setting.
- **Under matched settings,** the upload path and `Scanner.run` behave identically (parity test).

## Blocked, with the missing artifact and the claim left unverified

- **Full-page OCR on detected lines** (`run_full_pages.py`), for all splits and scanner on/off: needs the docTR detector weights, either `models/ocr/det_fast_base_ft.pt` or `~/.cache/doctr/models/fast_base-688a8b34.pt` of at least 1 kB. The host `doctr-static.mindee.com` is blocked, and the cached file here is a 0-byte failed download.
  - **Unverified:** page CER/WER on detected lines; missed, extra, split, merged and misordered lines; lost text from crops; scanner-on vs scanner-off; page p50/p95 latency.
  - PR #15 reported a full-page result from its own run (2.66 % CER on 30 holdout pages); it was not reproduced here.
- **Real, writer-disjoint pages** (handwriting, faint print, rotated phone photos): none are available. IAM and GNHK need `huggingface.co`, which is blocked.
  - **Unverified:** any accuracy claim on real use.

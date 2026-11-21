# Review 2026-11: benchmark results

Produced by the runners in `benchmarks/` (see `benchmarks/README.md`). `baseline_0c71106/` holds full runs at
`0c71106e54f4d2d3010c9e89e24b70d011f6fa9a` (main, PR #13), each run twice with identical output apart from
`generated_at` and latency fields. Smoke or partial runs go under `smoke/` with `_firstN` names. Every number
is computed on invented or generated data (or, for the page threshold only, the repository's stored per-photo
rows); none is an estimate.

## Baseline at 0c71106

### Medication warnings (`warnings.json`, 295 lines)

| Measure | misread option off (default) | misread option on |
|---|---|---|
| Original target: either flag on the original 39 numeric lines, `>= 0.95` | 37/39 = 0.9487, **FAIL** | 38/39 = 0.9744, PASS |
| (a) either flag on all numeric lines | 186/193 | 188/193 |
| (a) either flag on lines without numbers | 8/102 | 9/102 |
| (b) prominent warning, precision / recall vs dose wrong or lost | 42/105 / 42/65 | 45/109 / 45/65 |
| (b) prominent warning, precision / recall vs warranted | 57/105 / 57/80 | 60/109 / 60/80 |
| (c) prominent warning on correctly read print, v2 / v1 | 7/140 / 39/62 | 8/140 / 39/62 |
| (d) exact drug / value / unit of the shown text | 83/147 | 83/147 |
| Silent dose errors (dose wrong or lost, no prominent warning) | 23/65 | 20/65 |

### Alpha conversion (`alpha.json`)

One opaque pixel in the far corner switches `scanner/convert.py` from "composite on white" to "keep the stored
colour": text grey changes by 68 levels (antialiased text, alpha <= 200), text/paper contrast of a 50 %
overlay doubles (69 -> 139 on test photo 200000). On the 20 partial-alpha photos, page alpha 254 instead of
255 (outside 128) gives IoU >= 0.9 on 10/20 against 15/20. RGB reference: test 151/200 IoU >= 0.9 (found
189/200), OOD 30/80 (found 46/80). Runs without torch.

### Pages (`pages.json`)

False alarms (20 per kind): tiles 6/20, blotch 2/20, dots 1/20, fabric 1/20, OOD light 1/20, all others 0/20
(blank white / grey / paper / dark, noise, wood, solid, light, gradient, stripes, checker, granite); tiny
page-like squares of 32 and 48 px 2/6. Generated pages, raw: low contrast found 19/20 but IoU >= 0.9 0/20;
one corner cut 13/15, two corners cut 14/15, regular 17/20. Stored real rows: 0.60 chosen on DocCorner val
(600/603 well-outlined pages kept, 5/120 false alarms); once on test 672/678 kept, 16/239 false alarms;
SmartDoc 1263/1271 kept. The blank and noise guards need pixels and cannot be measured from stored rows.

### Handwriting (`handwriting.json`, synthetic, fonts seen in training: optimistic)

Test part (30 pages, 167 annotated lines): shipped rule line CER 0.0177, line exact 121/167, numeric lines
exact 93/133; handwriting reader alone 0.0244; printed reader alone 0.0702. Both readers wrong 36/167,
selection loss 10/167, routing loss 1/167. Decimal points dropped at confidence >= 0.9 (1.25 -> 125 at
0.91 and 0.93). Candidate margin selector (-0.05, chosen on validation) did not beat the shipped rule on test.

## BLOCKED

- Real held-out handwriting pages (GNHK, IAM): need huggingface.co.
- Automatic line detection, end-to-end page OCR, `python -m ocr.evaluate_v2`: need docTR's `fast_base` weights
  (`models/ocr/det_fast_base_ft.pt` or `~/.cache/doctr/models/fast_base-*.pt`; host doctr-static.mindee.com).
- Real page photos (DocCorner, SmartDoc): not available; only the stored per-photo rows are used.

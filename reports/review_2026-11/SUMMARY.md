# Review 2026-11: OCR warnings and page scanner, before / after

- **Before:** `0c71106` (main after PR #13). Results in `baseline_0c71106/`.
- **After:** `7b3a76d`. Results in `after_ce56e17/`, `after_52ccba3/`, `after_2d80375/` and `after_5d3a676/` (each an intermediate head of the review branch).
- All figures are n/N computed by the runners in `benchmarks/` on invented or generated data. Nothing is estimated.
- **Read the results in two separate parts:**
  - **Warning flags:** the "Check the numbers" warning, which changed.
  - **Transcription accuracy:** the text the readers produce, which did not change.

## Medication warnings

460 invented, labelled lines in `benchmarks/warnings/fixtures.jsonl`, in these sets: `original_39`, `extended`, `extended_r2` to `extended_r5`.

"Shipped default" is the setting each commit ships with:
- At `0c71106`, the misread-digit check is off, and v1 has no such check at all.
- At `7b3a76d`, it is on for both v1 and v2, controlled by the single switch `ocr.common.FLAG_MISREAD_DIGITS`.

| All 460 lines, shipped default | 0c71106 | 7b3a76d |
|---|---|---|
| Original-39 target, recall of either flag (`>= 0.95`, unrounded) | 37/39 = 0.9487, FAIL | 39/39, PASS |
| Prominent warning on lines whose dose is wrong or lost | 97/138 | 134/138 |
| Silent dose errors | 41/138 | 4/138 |
| Prominent warning on correctly read print, v2 | 34/198 | 2/198 |
| Prominent warning on correctly read print, v1 (by design: v1 flags any number or unit) | 57/92 | 53/92 |
| Any flag on lines without numbers | 43/161 | 33/161 |
| Drug, value and unit exactly right in the displayed text | 108/235 | 108/235 (unchanged) |

**Explicit off at `7b3a76d`:**
- target: 37/39, FAIL;
- silent dose errors: 28/138.

**Still silent with the shipped default (4):**
- "Ramipril 25 mg OD", "Lorazepam .5 mg", "Lorazepam 5 mg": plausible wrong numbers, which no text rule can catch.
- "Collected: 02/06/2026 O9:15".

**False warnings left on correctly read v2 print (2):**
- "So mg doses were reduced";
- "Result (mmol/L)" as a bare column header.

**Displayed and saved text is never altered.** Checked with `_line` on 1380 cases, and by the UI save-and-reopen step.

## Alpha conversion

- The one-pixel discontinuity is gone. Images that differ by one distant opaque pixel now convert identically: text difference was 68–104 grey levels and is now 0; outline IoU between the pair is now 1.0.
- The upload path now matches direct scanning, including with manual corners.
- **20 photos with page alpha 254:**
  - IoU ≥ 0.9: 10/20 → 15/20;
  - found: 18/20 → 19/20;
  - 3 photos are lower, and now equal the RGB reference.
- **Synthetic sets are unchanged:**
  - test: found 189/200, IoU ≥ 0.9 on 151/200, mean IoU 0.8867;
  - OOD: found 46/80, IoU ≥ 0.9 on 30/80, mean IoU 0.6727.
- **Still open:** an I;16 image holding only values 0..255 converts nearly black. It is documented and left as is, because scaling by the image's own maximum would reintroduce a one-pixel discontinuity.

## Page detection and geometry

Unchanged by the review.

- **False alarms**, 20 images per kind: tiles 6/20, blotch 2/20, dots 1/20, fabric 1/20, light desk 1/20; 0/20 on every blank and noise group.
- **Recall on generated pages:**
  - low-contrast pages: found 19/20, but IoU ≥ 0.9 on 0/20;
  - one corner cut off: 13/15 found;
  - two corners cut off: 14/15 found.
- **IoU ≥ 0.9 by input variant**, 70 photos each: raw 44, cropped 55, rot90 46, rot180 42, tilted 4° 40.
- **Threshold 0.60** stays the validation choice: 600/603 well-outlined pages kept, 5/120 false alarms.
  - Held-out test, evaluated once: DocCorner 672/678 kept with 16/239 false alarms; SmartDoc 1263/1271 kept.
  - These come from stored per-photo rows, so they cannot measure the blank and noise guards, which need the pixels.

## Handwriting

Synthetic, and optimistic because the fonts were used in training. 167 annotated test lines.

- **Where errors come from:**
  - recognition (both readers wrong): 36/167;
  - selection: 10/167;
  - routing: 1/167.
- **High-confidence errors:**
  - at confidence ≥ 0.9, 13/92 accepted lines are wrong;
  - decimal points are dropped, e.g. "1.25 mg" read as "125 mg" at 0.91–0.93.
- **Beam-search experiment**, opt-in and off by default (`HandwritingReader(num_beams=…)`, `Scanner(hw_beams=…)`):

  | Setting | Line CER | Exact lines | Handwriting latency per line |
  |---|---|---|---|
  | greedy | 0.0177 | 121/167 | 523 ms |
  | beams 2 | 0.0171 | 122/167 | 662 ms |
  | beams 4 | 0.0168 | 123/167 | 935 ms |

  It does not fix the dropped decimal points.

## Tests at 7b3a76d

| Suite | Result |
|---|---|
| Full environment | 536 passed, 0 failed, 6 skipped |
| Base environment | 475 passed, 0 failed, 17 skipped |
| UI | 40/40, no JavaScript errors |
| `OCR_SLOW=1` | the 2 slow tests skip with "model weights missing on this computer: detector" |

## Blocked, with exact prerequisites

- **Automatic line detection, end-to-end page OCR and `python -m ocr.evaluate_v2`:** need `~/.cache/doctr/models/fast_base-688a8b34.pt` (host doctr-static.mindee.com) or `models/ocr/det_fast_base_ft.pt`.
- **Real handwriting (GNHK, IAM):** needs huggingface.co.
- **The blank, noise and alpha guards on real photos:** need the DocCorner and SmartDoc images.

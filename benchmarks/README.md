# Benchmarks (review 2026-11)

Runners that score one checkout of this repository, so the same runner measures the baseline and later
commits. Each imports the code under test from `--root` (put first on `sys.path`; the result records the
commit and whether `ocr/`, `scanner/` or `server.py` had uncommitted edits) and writes `<out>/<name>.json`
plus a short `<name>.md`. Everything is seeded (2026), outputs are sorted, and the only time stamp is the
`generated_at` field. `--limit N` makes a smoke run that writes `<name>_firstN.json`, so it never overwrites a
full result; put smoke runs under `reports/review_2026-11/smoke/`.

All text and images are invented or generated here. No dataset image or text (FUNSD, SROIE, CORD, IAM, GNHK,
DocCorner, SmartDoc) is copied; the page benchmark reads only the repository's own stored per-photo result
rows (`models/finetune/eval/*.csv.gz`).

| Runner | Needs | What it measures |
|---|---|---|
| `run_warnings.py` | requirements.txt + requirements-ocr.txt (it imports `ocr.scan`); no model is loaded | "Check the numbers" flags (`ocr.scan.confirm_numbers`, `ocr.api._line`) on 295 labelled lines (`warnings/fixtures.jsonl`) |
| `run_alpha.py` | requirements.txt only | transparent and 16-bit images: paired-image discontinuity, `/api/scan` upload path, 20 partial-alpha photos, synthetic test / OOD |
| `run_pages.py` | requirements.txt only | false alarms by background kind, recall on stored and generated pages (low contrast, cut corners), crop / rotation / tilt, lost page and text area, stored real-photo threshold trade-off |
| `run_handwriting.py` | full OCR environment + `transformers` (requirements-handwriting.txt) | handwriting vs printed reader and the selection rule on annotated synthetic handwritten lines; calibration; candidate selector; line detection (BLOCKED without detector weights) |

## Running

```bash
# baseline checkout at a fixed commit, outside the shared worktree
git worktree add --detach /tmp/base 0c71106
python -m benchmarks.run_warnings    --root /tmp/base --out reports/review_2026-11/baseline_0c71106
python -m benchmarks.run_alpha       --root /tmp/base --out reports/review_2026-11/baseline_0c71106
python -m benchmarks.run_pages       --root /tmp/base --out reports/review_2026-11/baseline_0c71106
PYTHONPATH=<dir holding transformers 4.57.1> HF_HUB_OFFLINE=1 \
python -m benchmarks.run_handwriting --root /tmp/base --out reports/review_2026-11/baseline_0c71106

# a later commit, compared photo by photo with the baseline
python -m benchmarks.run_alpha --root . --out reports/review_2026-11/<sha> \
    --baseline reports/review_2026-11/baseline_0c71106/alpha.json

# two runs are identical apart from generated_at and latency fields:
python -m benchmarks.common A.json B.json
```

Compare handwriting runs made with the same `--threads`: torch's float results change in the fourth decimal
of a confidence between one and two threads (the baseline used `--threads 1`). `--hw-option KEY=VALUE`
(repeatable, e.g. `num_beams=4`) passes a decoding option to `HandwritingReader(...)` or its `read(...)`; a key
the checkout lacks is ignored and listed under `hw_options.ignored` in the result. `run_warnings --name
warnings_r2` writes the round-2 result without touching `warnings.json`.

Run from the repository root (the runners import `benchmarks.*` from the current directory and the code
under test from `--root`). The synthetic page photos (`data/synthetic/{test,ood}`) are not in git;
`run_alpha` and `run_pages` look in `--data`, then `<root>/data/synthetic`, then the main checkout. Limit
threads on a shared machine: the scanner runners set one thread (its OpenMP predictor is faster that way under
load), the others two.

## Medication-warning fixtures

`warnings/fixtures.jsonl` is written by `python -m benchmarks.warnings.build_fixtures` (a test checks that the
committed file matches). Each record has `id`, `set`, `category`, `path`, `routing`, `source_script`,
`ground_truth`, `displayed_text`, `handwriting_text` / `printed_text` (as the v2 Scanner stores them),
`has_numeric_content`, `dose_incorrect_or_lost`, `warning_warranted`, `expected_fields` and `origin`; the
builder's docstring defines each label.

`set: "original_39"` is exactly the 39 lines with a digit in the reference of the earlier evaluation's case
list (the "flagged by either flag: 37/39" figure), in the same order and routing; at 0c71106 the runner
recomputes 37/39. The rest (`set: "extended"`) are that list's lines without numbers, its two one-page lists
on v2 and v1, and new cases: a unit that lost its number next to a list index, a duration or another
medicine's dose; "Metformin SOO my" under every routing with the misread reading shown or kept as the
alternate; "InSulin O Units"; a dose kept only by the alternate reader; unit-word collisions; decimals;
missing quantities; mcg / mL / IU / mmol/L / g/dL; od / bd / tds / prn / nocte / twice daily; numbered lists;
printed values and dates; plain-text controls; v1 and v2.

`set: "extended_r2"` (76 records, ids `r2-001`...) was appended in round 2 without changing any earlier record:
count or frequency words without a digit, unit-before-value lab layouts and unit headers, a lower-case form
word before the medicine ("tab Metformin 500 mg"), v1 misreads ("Metformin SOO my", "Warfarin B my") and
look-alike v1 sentences, and misread units next to kept digits ("50 mcq", "10 mI", "11.8 q/dL"), each with
correct-print negatives and lost or misread positives on v1 and v2. In the results `set=all` stays the
round-1 sets (original_39 + extended), `set=extended_r2` is the new set and `set=all_with_r2` both; each mode
also has a per-category table.

Metrics, each with numerator / denominator and the list of misses, split by set and path, in three modes of the
misread-digit check: `shipped_default` (the setting the commit ships, read from its code), `explicit_off`
and `explicit_on` (results written before round 5 use `misread_option_off (default)` / `misread_option_on`,
in which v1 always ran as shipped): (a) numeric content detected by either flag; (b) precision and
recall of the prominent warning (`confirm_numbers`) against `dose_incorrect_or_lost` and against
`warning_warranted`; (c) prominent warnings on correctly read printed lines; (d) exact drug / value / unit
agreement (`fields.py`, whose docstring lists its limits) and silent dose errors; (e) the original target:
recall of "either flag" on the original 39, `>= 0.95`, PASS / FAIL without rounding.

The v2 line flags are set by a small adapter that does what `Scanner.run` does with routing on
(`confirm_numbers(ln, script_known=True[, misread_digits=True])`, then `has_numbers`); the result records the
function signatures it found and whether `Scanner.run` still makes that call, so an interface change is
visible.

## Handwriting data

Pages are rendered here, word by word, with the repository's OFL handwriting fonts (`data/raw/fonts`), so every
word and line has a known quad and text. Validation pages: seeds 0..19 (used only to choose the candidate
selector's margin); test pages: seeds 1000..1029 (evaluated once). The handwriting reader was fine-tuned on
lines rendered with the same fonts, so these numbers are optimistic. Real held-out pages (GNHK, IAM) need
huggingface.co and are not measured. Automatic line detection needs docTR's `fast_base` weights
(`models/ocr/det_fast_base_ft.pt` or `~/.cache/doctr/models/fast_base-*.pt`); `match_lines` (missed, extra,
split, merged, misordered) is tested with a fake detector in `tests/test_benchmarks.py`.

### Paired full-page handwriting OCR

Run the complete scanner and both annotated-crop readers on identical generated pixels in the same process:

```sh
python -m benchmarks.run_full_pages --root . --split regression --out /tmp/ocr-review
```

Requires the full offline OCR environment, all local checkpoints, and the cached FAST detector.
The regression split uses seeds 1000–1029, validation uses 0–19, and the additional synthetic holdout
uses 4100–4129. All three use the generator's bundled fonts, which also occurred in training.
Use `--limit N` for a separately named smoke result. Keep outputs from different commits in different directories.

`detected` receives the original image through the default Scanner. `annotated` supplies the labelled
line and word geometry to the same readers. Page scores collapse whitespace and preserve case and
punctuation. For line scores a monotone alignment first maximises exact anchors, then minimises character
edit costs between them. Extra output contributes insertions to CER and WER. Missing lines count as empty
predictions. The independent box comparison uses source coordinates and reports misses, extras, splits,
merges and order inversions. These geometry metrics need not match the text alignment.

The historical drug/value/unit score folds case and matches known names. Additional metrics report unit
case agreement, exact decimal values for reference drugs with a decimal dose, missing drug names, and
incomplete fields on recognised drugs. Neither parser checks all medical instructions. Each JSON includes
input hashes, model hashes, inference settings, software versions, per-page timings and aligned predictions.

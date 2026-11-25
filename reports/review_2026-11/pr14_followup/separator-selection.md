# Numeric separators and reader selection

**Decision. Keep the shipped reader selection.** Reusing the alternative reader helped on the validation and previously published regression pages, but introduced an extra error on new handwriting. Its effect on fresh whole-page CER and WER was zero.

## Experiment

Validation disagreements included a blood-pressure slash read as a digit and a decimal point omitted by both candidates. When the chosen handwriting reading and the printed reading have the same number of words, the experimental rule checks for a numeric token in the printed reading with an internal decimal point, decimal comma or slash. Removing the separator, or reading a slash as 1 or 7, must yield the corresponding handwriting token. Other words must agree after case folding within a validation-selected edit budget. The candidate is the complete, unmodified printed reading. When neither reader preserves the separator, this rule cannot recover it.

Only seeds 0–19 selected the budget. Zero context edits gave 53 character errors, one edit gave 52, and the shipped rule gave 54. We fixed the budget at one edit before scoring the old regression set, a new 30-page handwriting holdout, and 20 new printed controls. All pages use invented text. The handwriting fonts were present during model training.

| Input and sample size | CER, shipped → candidate | WER, shipped → candidate | Exact lines | Numeric lines exact | Number sequences exact on lines containing decimals |
|---|---:|---:|---:|---:|---:|
| Validation, 113 annotated lines | 2.25% → 2.17% | 7.44% → 7.00% | 86 → 87 | 71 → 72 of 96 | 20 → 20 of 21 |
| Existing regression, 167 annotated lines | 1.77% → 1.68% | 7.70% → 7.25% | 121 → 124 | 93 → 96 of 133 | 23 → 24 of 27 |
| New handwriting, 161 annotated lines | 2.39% → 2.49% | 10.77% → 10.93% | 110 → 109 | 85 → 84 of 134 | 24 → 24 of 30 |
| New print, 107 annotated lines | 0% → 0% | 0% → 0% | 107 → 107 | 96 → 96 of 96 | 26 → 26 of 26 |
| Same new handwriting, 30 full pages | 2.66% → 2.66% | 11.75% → 11.75% | | | |

On the fresh annotated pages, three lines changed. Two changes recovered a date separator while introducing a spelling error in the preceding word. The third change replaced the correct “BP 139/96” with “BP 13,9/96”. Exact drug/value/unit agreements remained 74/97. On the same whole-page outputs, only one change fired. It corrected a date but changed “Review” to “Reviel”. No printed control line was routed to the handwriting reader, so this control covers regressions on ordinary clean printed lines only.

The extra selection step averaged around 0.003 ms per fresh annotated line, measured by replaying both candidates and collecting the overrides. It needs no extra inference. The printed-control diagnostic ran each reader on every annotated line so its latency is separate from deployed OCR latency.

## Reproduction

From the repository root in the full OCR environment with the local model files:

```sh
export HF_HUB_OFFLINE=1
python -m benchmarks.run_separator_selection --threads 1 --printed-out /tmp/printed-selector-control.json
python -m benchmarks.run_full_pages --root . --split holdout --threads 2 --out /tmp/full-pages
python -m benchmarks.run_separator_selection --printed /tmp/printed-selector-control.json \
  --holdout /tmp/full-pages/full_pages_holdout.json --out /tmp/separator-selection.json
python -m pytest -q tests/test_separator_selection.py
```

The printed control uses DejaVu and Liberation fonts listed with SHA-256 hashes in the saved reader outputs. The JSON summary records source hashes, both validation settings, counts, latency and every changed annotated or whole-page reading. The paired runner stores the detector and recogniser checkpoint hashes. Runtime-dependent fields can differ on replay. [Reader outputs](printed-selector-control.json) · [Selector results](separator-selection.json)

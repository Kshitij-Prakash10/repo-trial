# Validation studies

These validation-split studies set three defaults in `ocr/scan.py` and `ocr/orientation.py`. No
test data was used to choose them.

| File | Script | What it decided |
|---|---|---|
| `orientation_flip_val.json` | `scripts/val_studies/flip_study.py` | the 180-degree check and its margin, `FLIP_MARGIN = 0.10` |
| `ablation_val.txt` | `scripts/val_studies/ablation_study.py` | which stage cost printed accuracy: deskew, then routing |
| `deskew_routing_val.json` | `scripts/val_studies/deskew_routing_study.py` | deskew off by default; the routing rule "handwriting reading only if more confident" |
| `deskew_border_val.txt` | (inline study, see the final report) | the deskew loss is not caused by how the rotated page's border is filled |

```bash
python scripts/val_studies/analyse.py
```

This prints every table from the JSON files without loading a model: the margin sweep (0.06 was best
but sat on a knife edge; 0.10 sits in the middle of the 0.08 to 0.15 plateau), deskew's CER change by
estimated angle, and the 19 routing rules.

The JSON files were made with the pipeline as it was before the fixes (handwriting reading always
used, deskew on). The scripts are updated to today's Scanner, so a rerun can differ slightly from
the committed numbers.

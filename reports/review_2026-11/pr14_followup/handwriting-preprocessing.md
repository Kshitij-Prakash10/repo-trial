# Handwriting preprocessing follow-up

A line with very sparse ink could become entirely black when the second percentile of its background-corrected pixels was already white. The reader now skips contrast stretching in this case. The regression fixture checks white paper and preservation of a sparse mark. All 441 annotated crops from validation, regression and holdout produced identical normalized pixels before and after the fix. Three full-page smoke scans also reproduced the previous OCR text, warnings and layout.

## Recognition experiment

The experiment uses the same annotated crops, TrOCR checkpoint, printed outputs, routing and reader selection for every variant. It compares the selected transcriptions after changing only the pixels fed to TrOCR. In each split, baseline and candidate use the same page batching. Previously reported per-line runs used a different batching scheme.

| Synthetic split and preparation | Line CER | Line WER | Exact lines | Exact numeric lines | Exact drug, value and unit | Exact number sequences on decimal lines |
|---|---:|---:|---:|---:|---:|---:|
| Validation baseline | 2.29% | 7.22% | 86/113 | 71/96 | 64/76 | 20/21 |
| Validation Lanczos | 2.25% | 7.22% | 85/113 | 70/96 | 64/76 | 20/21 |
| Validation light stroke strengthening | 2.34% | 7.44% | 85/113 | 70/96 | 64/76 | 20/21 |
| Holdout baseline | 2.42% | 11.26% | 107/161 | 83/134 | 73/97 | 26/30 |
| Holdout Lanczos | 2.58% | 11.42% | 105/161 | 82/134 | 73/97 | 26/30 |

Stroke strengthening increased both validation error rates. Lanczos saved one character edit on validation while losing an exact numeric line. On the holdout it increased character and word errors and changed a previously correct date from `16/09/2026` to `16,000/2026`. Both methods remain benchmark-only. CPU timings from this run share the machine with verification jobs and cannot establish a latency improvement.

The synthetic pages use fonts seen in training. The pinned IAM test download returned HTTP 403 in this environment. Real-writer accuracy and the prevalence of sparse crops on real pages remain unmeasured. [Reproduction commands](../../../benchmarks/README.md#handwriting-preprocessing-comparison)

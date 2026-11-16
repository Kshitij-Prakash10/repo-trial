# Datasets used by the handwriting and orientation upgrade

Nothing inside a download is executed; archives are only read. English only.

## In this repository

The datasets whose licences allow redistribution are meant to be stored with Git LFS. **That upload has
not been done yet**, so today `git lfs pull` brings nothing. They are added by running
`scripts/fetch_shareable_data.sh` once on a machine whose git can upload LFS files; if the paths below
are missing in your clone, that upload has not been done yet and the same script fetches the data.
`python scripts/shareable_manifest.py` verifies the files against `data/shareable.sha256.json`.

| Path | Dataset | Licence | Size |
|---|---|---|---|
| `data/raw/gnhk/GNHK-dataset.zip` | GNHK, as published (official train and test pages with annotations); unpacked on first use into `data/raw/gnhk/pages/` | CC BY 4.0 | 1.0 GB |
| `data/sroie/` | SROIE receipts, converted: `images/<split>/`, `<split>.jsonl`, `splits.json` | CC BY 4.0 | 0.56 GB |
| `data/cord/` | CORD v2 receipts, converted the same way | CC BY 4.0 | 2.2 GB |
| `data/raw/fonts/` | 24 handwriting fonts; the licence text for each font family is in `data/raw/fonts/licences/` | SIL Open Font License 1.1 | 3 MB, plain git (no LFS) |

Not in the repository (licences do not allow re-sharing): **IAM** (non-commercial research only,
registration required) and **FUNSD** (non-commercial research and education only). Download them with
`python scripts/download_handwriting.py --only iam` and `python scripts/download_data.py --only funsd`.
Both download scripts skip data that is already present.

Attribution (CC BY 4.0):
- **GNHK**: GoodNotes, "GNHK: A Dataset for English Handwriting in the Wild", Lee et al., ICDAR 2021.
  Copied unchanged from the Hugging Face mirror `staghado/GNHK-Dataset` @ `eaa67d39`.
- **SROIE**: ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction,
  Huang et al. Converted from `jsdnrs/ICDAR2019-SROIE` (see `NOTES.md`, Step 3). Images unchanged;
  annotations reformatted to one JSON line per page.
- **CORD**: NAVER Clova, "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing", Park et al., 2019.
  Converted from `naver-clova-ix/cord-v2` in the same way.

| Dataset | Source (pinned) | Licence | Size on disk | Splits used |
|---|---|---|---|---|
| IAM handwriting, text lines | Hugging Face `Teklia/IAM-line` @ `fbdad975` | The mirror is tagged MIT. The original IAM database is free for non-commercial research only (registration at the official site); treat the data as research-only. | 0.27 GB | Official writer-independent (RWTH / Aachen) split as shipped: train 6,482, validation 976, test 2,915 lines |
| GNHK (GoodNotes Handwriting Kollection) | Hugging Face `staghado/GNHK-Dataset` @ `eaa67d39`, file `GNHK-dataset.zip` only | CC BY 4.0 (GoodNotes; the mirror itself carries no licence tag) | 1.0 GB zip + 1.0 GB unpacked | Official train 515 pages and test 172 pages. No official validation split: 52 training pages are set aside (seed 2026); 1 training page that is byte-identical to a validation page is dropped. Train 462 / val 52 / test 172 pages |
| Handwriting fonts (for synthetic lines) | `github.com/google/fonts`, `ofl/` folder, 24 fonts; SHA-256 of each file in `data/fonts.sha256.json` | SIL Open Font License 1.1 | 3 MB, plain git (no LFS) | training only |
| FUNSD, SROIE, CORD v2 (printed) | see `NOTES.md`, Step 3 (`scripts/download_data.py`) | FUNSD research-only; SROIE and CORD CC BY 4.0 | 2.8 GB | unchanged; official test splits |

Synthetic handwriting lines are generated on the fly (`ocr/synth_hw.py`) from the fonts above and
from training transcriptions only. They are synthetic and are used for training only.

## Skipped

| Dataset | Why |
|---|---|
| CVL Database | Hosted at TU Wien / Zenodo, both unreachable from this machine; no Hugging Face copy found. |
| Imgur5K | The official release is a list of Imgur links under a non-commercial licence (many links are dead); the Hugging Face copies are unofficial re-uploads with no licence or marked "other / manual". Not used. |
| IAM from the official site | Needs a registered account. The Hugging Face mirror above was used instead. |
| Other files in `staghado/GNHK-Dataset` | The repository also holds Imgur5K and a 25 GB bundle of unknown origin; only `GNHK-dataset.zip` is downloaded. |

## Leakage checks (run on 2026-11-14)

- IAM: no image appears in two splits (MD5 of the image bytes; 0 duplicates across train / validation / test).
  The mirror does not ship writer ids, so writer independence rests on it being the official split
  (line counts 6,482 / 976 / 2,915 match the published RWTH split); it could not be checked per writer.
- GNHK: no page image appears in two splits after dropping the one duplicate noted above.
- The pretrained handwriting model (`microsoft/trocr-small-handwritten`) was itself fine-tuned on the
  IAM training set by its authors; IAM test lines are from different writers.

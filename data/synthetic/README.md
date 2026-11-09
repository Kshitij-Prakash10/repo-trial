# Synthetic scanner dataset (SYNTHETIC, not real photos)

Every image in this folder is **synthetic**: a computer-rendered health document (prescription, lab
report, medical certificate, discharge summary, insurance claim form; the `ood` split also has referral
letters, pharmacy bills and blood-sugar logs) pasted with a random perspective onto a rendered background,
then lit, shadowed, blurred, noised and JPEG-compressed. All names, words and values on the pages are
invented. No real photo and no real person's data is in this folder.

The page detector in `models/scanner.joblib` is trained on `train/` and evaluated on `val/`, `test/`
and `ood/` (see `models/scanner_metrics.json` and `docs/scanner.md`).

## Splits

| Split | Photos | Generator seeds | Generator style | Use |
|---|---|---|---|---|
| `train/` | 360 | 0 to 359 | `train` | training |
| `val/` | 60 | 100000 to 100059 | `train` | validation (design choices) |
| `test/` | 200 | 200000 to 200199 | `train` | held-out test |
| `ood/` | 80 | 300000 to 300079 | `ood` | held-out, different generator setting never used in training (other layouts, chequered and granite backgrounds, stronger tilt, warm light) |

Seed ranges are disjoint, so no evaluation photo, background or page is shared with training.

## Files

- `<split>/<seed>.jpg`: the photo, JPEG quality 90, stored at the full generated resolution (long side
  480 to 1280 px). Nothing was downscaled or sharded: the whole folder is about 71 MB and the largest file
  is well under 1 MB.
- `<split>/labels.csv`, one row per photo:
  - `filename`, `seed`, `style` (generator style), `width`, `height` (px)
  - `tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y`: the 4 true page corners in pixels, in the order
    top-left, top-right, bottom-right, bottom-left **of the page content** (so a page lying sideways has its
    top-left corner on a side of the photo). Origin at the top-left of the image; a pixel covers
    [x, x + 1). Corners can lie slightly outside the image when the frame cuts the page off.
  - `kind` (page layout), `background` (background type), `cut` (1 when the frame cuts part of the page off)
  - `sha256` of the JPEG file
- The page mask is the polygon of the 4 corners (clipped to the image); it is not stored separately.

## How it was generated

Generator: `scanner/synth.py`, version `synth-1.0` (`scanner.dataset.GENERATOR_VERSION`), seeded with
`numpy.random.default_rng([seed, style])`; storage: `scanner/dataset.py` (JPEG quality 90 with Pillow).
Built with Python 3.13, numpy 2.5.4, Pillow 12.3.0 (libjpeg-turbo 6.2 API).

Regenerate every file:

```bash
python -m scanner.dataset --force    # writes data/synthetic/<split>/ and labels.csv
python -m scanner.dataset --verify   # checks every file against its sha256
```

With the same numpy, Pillow and libjpeg build the output is byte-for-byte identical (a test in
`tests/test_scanner.py` regenerates one photo per split and compares its sha256). Other Pillow or libjpeg
versions can give slightly different JPEG bytes; the corners do not change.

`python -m scanner.train` trains from this folder and regenerates it first when it is missing.

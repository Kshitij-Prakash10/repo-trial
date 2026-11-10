# Health Diary

A basic web app that works as a personal health diary. Each user keeps their
**current** health status and their **past** health records in one place.

## Features

- **Multiple users**: create a user, switch between users, delete a user.
- **Dashboard**: the current snapshot (latest height, weight, BMI, blood
  pressure, heart rate, blood group), active conditions, current medications,
  and a timeline of recent records.
- **Vitals**: dated entries for height, weight, BP, heart rate, blood sugar.
  BMI is worked out automatically.
- **Diseases & Conditions**: name, diagnosis date, status (Active / Chronic / Resolved).
- **Medications**: dosage and start/end dates. A medication with no end date,
  or an end date that hasn't passed, counts as current.
- **Allergies**: allergen, severity, reaction.
- **Visits & Reports**: doctor visits, lab tests, vaccinations, surgeries.
- **Profile**: name, date of birth (age is worked out from it), gender, blood
  group, emergency contact.
- **Documents**: store prescriptions, lab reports, medical certificates,
  discharge summaries, insurance papers and other files (PDF, PNG, JPEG, WebP;
  up to 20 MB) per user, with title, category, date, tags, notes and an
  optional link to a visit or condition. Search, filter, view, download, edit,
  delete. The latest documents show on the Dashboard.
- **Scan**: its own view (sidebar, Dashboard quick action, or Documents) in three
  steps, Photo, Check and Save. Take or choose a photo (or drop one); the local
  scanner finds the page with a trained machine-learning model, straightens it
  and cleans it up (colour, grey or black-and-white). Drag the corners (or move
  them with the arrow keys) and Apply if the outline is off. Save it with a title,
  category, date, tags and notes; it appears in Documents. See `docs/scanner.md`.
- **Voice notes**: record in the browser (or upload an audio file), attach to a
  document, play, rename, delete.
- Storage used per user is shown in the sidebar and on the Profile page.
  Deleting a user deletes their documents and voice notes. Light and dark
  theme; works down to 360 px wide.

## Running

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python server.py            # http://127.0.0.1:8000  (HOST and PORT env vars to change)
```

Opening `index.html` directly also works; only scanning needs the server.

### Scanning a document

1. Open **Scan** (or "Scan a document" on the Dashboard). The small print at the
   bottom names the page-detection model in use.
2. **Photo**: take a photo, choose a file or drop one (JPEG, PNG or WebP, up to
   10 MB). Put the page on a dark or contrasting background, get the whole page
   in the frame, and use good light.
3. **Check**: compare the outline on your photo with the corrected page; switch
   Colour / Grey / Black & white. If the outline is wrong, choose Adjust corners,
   drag the four handles (or Tab to one and use the arrow keys, Shift for bigger
   steps) and Apply. If the page edges were not found, retake, set the corners,
   or save anyway.
4. **Save**: give it a title, category, date, tags and notes, then "View in
   Documents" or "Scan another".

Without the server (opened as a file) or without the scanner, the Scan view
still saves the photo as a document, without scanning.
Recording audio needs http://127.0.0.1, http://localhost or https.
Everything stays in the browser (localStorage for records, IndexedDB for
files); nothing needs the internet at run time.

## OCR scanner (text from a photo or scan)

`ocr_scan.py` turns a photo or scan of a document into text, with a box and a confidence for every
word and line. It runs on the CPU. Scope: English printed text.

```
input image -> page scanner (existing model, unchanged) -> text detection -> text recognition -> ordered text
```

The page scanner's crop is used only when it is safe (page found, confident, and no detected text
outside it); otherwise the whole image is read. Design notes and the reasons for each choice are in
`NOTES.md`; measured results are in `RESULTS.md`.

### Setup

```bash
pip install -r requirements-ocr.txt
```

```bash
sudo apt-get install -y tesseract-ocr
```

Tesseract is needed only for the baseline comparison. The first run downloads the docTR pretrained
weights (about 60 MB) into `~/.cache/doctr`; after that everything runs offline.

Limits: pages turned by 90 or 180 degrees are not corrected (rotate them first; EXIF rotation is
applied), and handwriting is out of scope.

### Run inference

```bash
python ocr_scan.py --image path.jpg --out result.json
```

This writes `result.json` and `result.txt` (the plain text). Options: `--weights pretrained` (docTR
weights without fine-tuning), `--no-scanner` (skip the page scanner stage), `--overlay boxes.png`
(the page with the word boxes drawn).

`result.json` holds: `text`; `lines` (`text`, `box`, `confidence`, `words` as indices); `words`
(`text`, `confidence`, `detection_score`, `box` on the processed page, `source_quad` in the pixels of
the input image, `line`); `scanner` (whether the crop was used and why); `models` (which stage uses
fine-tuned weights). Boxes are `[x0, y0, x1, y1]` in pixels.

### Download the data

```bash
python scripts/download_data.py
```

Downloads FUNSD, SROIE and CORD v2 from Hugging Face (pinned revisions, about 6 GB on disk) into
`data/<dataset>/` and converts them to one annotation format with fixed train / val / test splits.
The data is not committed. The script checks free disk space first and refuses to pass 30 GB.

### Train

```bash
python -m ocr.train_rec --config configs/ocr/rec.json --overfit 48
```

```bash
python -m ocr.train_rec --config configs/ocr/rec.json
```

The first command is the tiny overfit test; the second fine-tunes the recogniser (about 1 hour on
4 CPU cores) and writes `runs/rec/best.pt`, `last.pt` and `log.json` (train loss and validation
CER / WER per epoch). Copy `best.pt` to `models/ocr/rec_crnn_mobilenet_v3_large_ft.pt` to use it.

```bash
python -m ocr.train_det --config configs/ocr/det.json --overfit 12
```

The detector's training script exists, but its overfit test does not pass yet (see `RESULTS.md`), so
no fine-tuned detector is shipped and the pipeline uses the pretrained one.

### Evaluate

```bash
python -m ocr.evaluate --system pretrained
```

```bash
python -m ocr.evaluate --system tesseract
```

```bash
python -m ocr.evaluate --system finetuned
```

Each run scores recognition (CER / WER), detection (precision / recall / F1 at IoU 0.5) and the full
page end to end (CER / WER) on the test split of every dataset, and writes
`runs/eval/<system>_test.json` plus per-item files. OCR tests: `python -m pytest -q tests/test_ocr.py`.

## Tests

```bash
python -m pytest -q                     # server and scanner tests (fine-tuning tests need requirements-finetune.txt)
npm install playwright && npx playwright install chromium   # once
node tests/ui/check.js                  # UI flows in headless Chromium (starts server.py)
```

## Project structure

```
index.html          Page layout: user selection screen + app (sidebar + views)
css/styles.css      Styling, responsive layout, light and dark theme
js/storage.js       Health records (localStorage, one record per user)
js/files-db.js      Documents and voice notes (IndexedDB; the only file that touches it)
js/ui.js            Shared UI helpers (formatting, icons, toasts, file viewer)
js/app.js           Rendering and event handling
js/documents.js     Document library
js/scan.js          Scan view: Photo, Check (corners), Save (talks to /api/scan)
js/voice.js         Voice notes (MediaRecorder)
server.py           Local Flask server: the static app + the scanner API
scanner/            ML document scanner (training, inference, API); docs/scanner.md
data/synthetic/     The synthetic photos the scanner is trained and tested on
models/             Scanner models (synthetic, and fine-tuned on real photos), metrics, ONNX export
docs/finetune/      Fine-tuning on real photos: results, ablations, failure cases
tests/              pytest tests and the Playwright UI check
```

## Notes

- Data is stored in the browser (`localStorage` and IndexedDB), so it stays on
  that one device and browser. There is no password protection yet.
- `js/storage.js` is the only file that touches storage. You can replace it
  with calls to a backend API (plus real login) without changing the UI.

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

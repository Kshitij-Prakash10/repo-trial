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
- **Scan a paper document**: take or choose a photo; the local scanner finds the
  page with a trained machine-learning model, straightens it and cleans it up
  (colour, grey or black-and-white); save it as a document. See `docs/scanner.md`.
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
Recording audio needs http://127.0.0.1, http://localhost or https.
Everything stays in the browser (localStorage for records, IndexedDB for
files); nothing needs the internet at run time.

## Tests

```bash
python -m pytest -q                     # server and scanner tests
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
js/scan.js          Paper document scan flow (talks to /api/scan)
js/voice.js         Voice notes (MediaRecorder)
server.py           Local Flask server: the static app + the scanner API
scanner/            ML document scanner (training, inference, API); docs/scanner.md
data/synthetic/     The synthetic photos the scanner is trained and tested on
models/             Trained scanner model and its held-out metrics
tests/              pytest tests and the Playwright UI check
```

## Notes

- Data is stored in the browser (`localStorage` and IndexedDB), so it stays on
  that one device and browser. There is no password protection yet.
- `js/storage.js` is the only file that touches storage. You can replace it
  with calls to a backend API (plus real login) without changing the UI.

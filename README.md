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

## Running

No build step and no packages to install. With Node.js 14 or later:

```bash
npm start
```

Then visit http://localhost:3000. To use another port, run `PORT=8080 npm start`.

You can also just open `index.html` in a browser, or serve the folder with
`python3 -m http.server 8000`.

## Project structure

```
index.html        Page layout: user selection screen + app (sidebar + views)
css/styles.css    Styling and responsive layout
js/storage.js     Data layer (localStorage, one record per user)
js/app.js         Rendering and event handling
server.js         Small Node.js static server (used by `npm start`)
package.json      npm scripts
```

## Notes

- Data is stored in the browser's `localStorage`, so it stays on that one
  device and browser. There is no password protection yet.
- `js/storage.js` is the only file that touches storage. You can replace it
  with calls to a backend API (plus real login) without changing the UI.

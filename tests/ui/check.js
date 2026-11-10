#!/usr/bin/env node
/*
 * Health Diary UI check: drives the main flows in headless Chromium with Playwright.
 *
 *   node tests/ui/check.js
 *
 * Starts `python server.py` on a free local port (or uses BASE_URL if set), then checks: an older saved
 * user still opens, create user, add vitals, add / search / edit / view documents (PNG, PDF, a rejected
 * fake file), voice note from an uploaded WAV (rename, play, attach to a document), recording with a fake
 * microphone, the Scan view (Photo, Check, Save) against a mocked /api/scan: Dashboard and Documents entry
 * points, outline and confidence, mode switch, dragging and keyboard-moving corners then Apply (sends the
 * `corners` field), page-not-found help, save and the scan listed in Documents; the real scanner (also with
 * manual corners) when the server reports it available; reading text against a mocked /api/ocr (Read text
 * after a scan, progress, the text panel, word count, Copy, low-confidence / no-text notes, inline error,
 * edited text saved with the document, the Text indicator, search and the viewer in Documents, an older
 * record without text, no Read text for a PDF, the actions disabled when /api/ocr/status says unavailable)
 * and once against the real OCR when the server reports it available; scanner missing and file:// (save the photo without
 * scanning); storage shown, deleting a user deletes their files, no horizontal scroll at 360 px, keyboard
 * menu on mobile, no page errors.
 *
 * Env: BASE_URL (skip starting a server), PYTHON (default "python3"), UI_PORT (default 8123),
 *      SCREENSHOT_DIR (save screenshots at 1280 and 375 px, light and dark).
 * Exit code 0 when every check passes. All test data is synthetic.
 */
"use strict";

const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const http = require("http");

const ROOT = path.resolve(__dirname, "..", "..");
const PORT = Number(process.env.UI_PORT || 8123);
let BASE = process.env.BASE_URL || `http://127.0.0.1:${PORT}`;
BASE = BASE.replace(/\/+$/, "");
const SHOTS = process.env.SCREENSHOT_DIR || "";

const results = [];
async function step(name, fn) {
  const t = Date.now();
  try {
    const note = await fn();
    results.push({ name, ok: true });
    console.log(`PASS  ${name}${note ? ` (${note})` : ""}  [${Date.now() - t} ms]`);
  } catch (err) {
    results.push({ name, ok: false, err });
    console.log(`FAIL  ${name}\n      ${String(err && err.stack || err).split("\n").slice(0, 14).join("\n      ")}`);
  }
}
const clearToasts = (p) => p.evaluate(() => document.querySelector("#toasts").replaceChildren());
function assert(cond, msg) { if (!cond) throw new Error(msg || "assertion failed"); }

/* ---------- synthetic test files ---------- */
function wavBuffer(seconds = 1.2, rate = 8000) {
  const n = Math.round(seconds * rate);
  const buf = Buffer.alloc(44 + n * 2);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write("WAVE", 8);
  buf.write("fmt ", 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(rate, 24); buf.writeUInt32LE(rate * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36); buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) buf.writeInt16LE(Math.round(8000 * Math.sin(2 * Math.PI * 440 * i / rate)), 44 + i * 2);
  return buf;
}
const PDF = Buffer.from("%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n");

// PNG made in the page with a canvas: a synthetic "document" (white page with lines) on a dark table.
async function pngFromPage(page, { w = 600, h = 800, photo = false } = {}) {
  const b64 = await page.evaluate(({ w, h, photo }) => {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const x = c.getContext("2d");
    x.fillStyle = photo ? "#3b3026" : "#ffffff";
    x.fillRect(0, 0, w, h);
    x.save();
    if (photo) { x.translate(w / 2, h / 2); x.rotate(0.12); x.translate(-w * 0.32, -h * 0.4); }
    const pw = photo ? w * 0.64 : w, ph = photo ? h * 0.8 : h;
    x.fillStyle = "#fbfaf6"; x.fillRect(0, 0, pw, ph);
    x.fillStyle = "#222";
    x.font = `${Math.round(pw / 18)}px sans-serif`;
    x.fillText("SYNTHETIC LAB REPORT", pw * 0.08, ph * 0.1);
    for (let i = 0; i < 14; i++) x.fillRect(pw * 0.08, ph * (0.18 + i * 0.05), pw * (0.4 + (i % 3) * 0.15), Math.max(2, ph / 160));
    x.restore();
    return c.toDataURL("image/png").split(",")[1];
  }, { w, h, photo });
  return Buffer.from(b64, "base64");
}

/* ---------- server ---------- */
function waitFor(url, ms = 15000) {
  const end = Date.now() + ms;
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      http.get(url, (r) => { r.resume(); resolve(); }).on("error", () => {
        if (Date.now() > end) reject(new Error(`server did not start at ${url}`));
        else setTimeout(tryOnce, 200);
      });
    };
    tryOnce();
  });
}

async function main() {
  let server = null;
  if (!process.env.BASE_URL) {
    const py = process.env.PYTHON || "python3";
    server = spawn(py, ["server.py"], { cwd: ROOT, env: { ...process.env, HOST: "127.0.0.1", PORT: String(PORT) }, stdio: ["ignore", "pipe", "pipe"] });
    let log = "";
    server.stdout.on("data", (d) => { log += d; });
    server.stderr.on("data", (d) => { log += d; });
    server.on("exit", (code) => { if (code) console.log(`server exited with ${code}\n${log}`); });
    await waitFor(BASE + "/");
  }
  if (SHOTS) fs.mkdirSync(SHOTS, { recursive: true });
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "hd-ui-"));

  const browser = await chromium.launch();
  const pageErrors = [];
  const newPage = async (opts = {}) => {
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, ...opts });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => pageErrors.push(`${page.url()}: ${e.message}`));
    page.on("console", (m) => { if (m.type() === "error" && !/Failed to load resource/.test(m.text())) pageErrors.push(`console: ${m.text()}`); });
    page.on("dialog", (d) => d.accept());
    return { ctx, page };
  };

  const { ctx, page } = await newPage();
  let scanRequests = [];

  await step("older saved user (original localStorage shape) still opens", async () => {
    await page.goto(BASE + "/");
    await page.evaluate(() => {
      const legacy = {
        id: "legacy1", createdAt: "2025-01-01T00:00:00.000Z",
        profile: { name: "Legacy Person", dob: "1980-05-05", gender: "Female", bloodGroup: "O+" },
        vitals: [{ id: "v1", date: "2025-01-02", height: "160", weight: "70", bp: "130/85", heartRate: "72", sugar: "", notes: "" }],
        conditions: [{ id: "c1", name: "Asthma", diagnosed: "2010-01-01", status: "Chronic", notes: "" }],
        medications: [], allergies: [], visits: [],
      };
      localStorage.setItem("healthDiary.users", JSON.stringify({ legacy1: legacy }));
      localStorage.removeItem("healthDiary.currentUser");
    });
    await page.reload();
    await page.click('[data-user="legacy1"]');
    await page.waitForSelector("#app:not(.hidden)");
    const cards = await page.textContent("#current-cards");
    assert(cards.includes("70 kg") && cards.includes("130/85"), "legacy vitals not shown: " + cards);
    assert((await page.textContent("#dash-conditions")).includes("Asthma"), "legacy condition missing");
    await page.click("#logout-btn");
    await page.waitForSelector("#auth-screen:not(.hidden)");
  });

  if (SHOTS) { await clearToasts(page); await page.screenshot({ path: path.join(SHOTS, "auth-1280-light.png"), fullPage: true }); }

  await step("create a user", async () => {
    await page.fill('#new-user-form [name="name"]', "Asha Test");
    await page.fill('#new-user-form [name="dob"]', "1990-04-12");
    await page.selectOption('#new-user-form [name="bloodGroup"]', "B+");
    await page.click('#new-user-form button[type="submit"]');
    await page.waitForSelector("#app:not(.hidden)");
    assert((await page.textContent("#user-name")) === "Asha Test");
    assert(await page.isVisible('[data-section="dashboard"]'));
  });

  await step("add vitals; dashboard shows latest values and BMI", async () => {
    await page.click('a[data-view="vitals"]');
    await page.waitForSelector('[data-section="vitals"]:not(.hidden)');
    await page.fill('[data-form="vitals"] [name="height"]', "170");
    await page.fill('[data-form="vitals"] [name="weight"]', "65");
    await page.fill('[data-form="vitals"] [name="bp"]', "118/76");
    await page.fill('[data-form="vitals"] [name="heartRate"]', "64");
    await page.click('[data-form="vitals"] button');
    await page.waitForSelector('[data-table="vitals"] tbody tr td[data-label="Weight"]');
    await page.click('a[data-view="dashboard"]');
    const cards = await page.textContent("#current-cards");
    assert(cards.includes("65 kg") && cards.includes("22.5") && cards.includes("118/76"), cards);
  });

  await step("add a visit and a medication (existing features)", async () => {
    await page.click('a[data-view="visits"]');
    await page.fill('[data-form="visits"] [name="date"]', "2026-03-02");
    await page.fill('[data-form="visits"] [name="doctor"]', "City Clinic");
    await page.selectOption('[data-form="visits"] [name="type"]', "Lab test");
    await page.click('[data-form="visits"] button');
    await page.waitForSelector('[data-table="visits"] td[data-label="Doctor / Hospital"]');
    await page.click('a[data-view="medications"]');
    await page.fill('[data-form="medications"] [name="name"]', "Metformin");
    await page.fill('[data-form="medications"] [name="dosage"]', "500 mg");
    await page.click('[data-form="medications"] button');
    await page.waitForSelector('[data-table="medications"] td[data-label="Medicine"]');
    assert((await page.textContent("#dash-medications")).includes("Metformin"));
  });

  const docPng = await pngFromPage(page, { w: 600, h: 800 });
  const photoPng = await pngFromPage(page, { w: 900, h: 1200, photo: true });
  fs.writeFileSync(path.join(tmp, "lab-report.png"), docPng);
  fs.writeFileSync(path.join(tmp, "prescription.pdf"), PDF);
  fs.writeFileSync(path.join(tmp, "not-really.pdf"), "hello, this is plain text");
  fs.writeFileSync(path.join(tmp, "note.wav"), wavBuffer());
  fs.writeFileSync(path.join(tmp, "photo.png"), photoPng);

  await step("documents: empty state, add a PNG with metadata and a link to the visit", async () => {
    await page.click('a[data-view="documents"]');
    await page.waitForSelector("#doc-list .state");
    assert((await page.textContent("#doc-list")).includes("No documents yet"));
    await page.click("#doc-add-btn");
    await page.setInputFiles("#doc-file", path.join(tmp, "lab-report.png"));
    assert((await page.inputValue("#up-title")) === "lab-report", "title not prefilled from file name");
    await page.fill("#up-title", "Blood test March");
    await page.selectOption("#up-category", "lab_report");
    await page.fill("#up-date", "2026-03-02");
    await page.fill("#up-tags", "blood, cbc");
    await page.selectOption("#up-link", { index: 1 });
    await page.fill("#up-notes", "Synthetic test document");
    await page.click("#doc-upload-save");
    await page.waitForSelector('.doc-card .doc-title:text("Blood test March")');
    const card = page.locator(".doc-card", { hasText: "Blood test March" });
    assert(await card.locator(".thumb img").count() === 1, "no thumbnail");
    assert((await card.textContent()).includes("Visit: Lab test"), "visit link not shown");
    assert(await page.isHidden("#doc-upload"), "upload form still open");
  });

  await step("documents: PDF accepted, fake file rejected by its first bytes", async () => {
    await page.click("#doc-add-btn");
    await page.setInputFiles("#doc-file", path.join(tmp, "not-really.pdf"));
    await page.click("#doc-upload-save");
    await page.waitForFunction(() => document.querySelector("#doc-upload-status").textContent.includes("not a PDF"));
    await page.setInputFiles("#doc-file", path.join(tmp, "prescription.pdf"));
    await page.fill("#up-title", "Prescription Dr Rao");
    await page.selectOption("#up-category", "prescription");
    await page.click("#doc-upload-save");
    await page.waitForSelector('.doc-card .doc-title:text("Prescription Dr Rao")');
    assert(await page.locator(".doc-card").count() === 2);
  });

  await step("documents: search and category filter", async () => {
    await page.fill("#doc-q", "cbc");
    await page.waitForFunction(() => document.querySelectorAll(".doc-card").length === 1);
    assert((await page.textContent("#doc-count")).includes("1 of 2"));
    await page.fill("#doc-q", "");
    await page.selectOption("#doc-cat", "insurance");
    await page.waitForSelector('#doc-list .state:has-text("Nothing matches")');
    await page.selectOption("#doc-cat", "prescription");
    await page.waitForFunction(() => document.querySelectorAll(".doc-card").length === 1);
    await page.selectOption("#doc-cat", "");
    await page.waitForFunction(() => document.querySelectorAll(".doc-card").length === 2);
  });

  await step("documents: edit, view and download", async () => {
    const card = page.locator(".doc-card", { hasText: "Prescription Dr Rao" });
    await card.locator('[data-act="edit"]').click();
    await card.locator('input[name="title"]').fill("Prescription Dr Rao (edited)");
    await card.locator('input[name="tags"]').fill("diabetes");
    await card.locator('button[type="submit"]').click();
    await page.waitForSelector('.doc-card .doc-title:text("Prescription Dr Rao (edited)")');
    const img = page.locator(".doc-card", { hasText: "Blood test March" });
    await img.locator('.card-actions [data-act="view"]').click();
    await page.waitForSelector("#viewer[open] #viewer-body img");
    if (SHOTS) { await clearToasts(page); await page.screenshot({ path: path.join(SHOTS, "viewer-1280-light.png") }); }
    const [dl] = await Promise.all([page.waitForEvent("download"), page.click("#viewer-download")]);
    assert(dl.suggestedFilename() === "lab-report.png", "download name " + dl.suggestedFilename());
    await page.keyboard.press("Escape");
    await page.waitForSelector("#viewer:not([open])", { state: "attached" });
  });

  await step("dashboard lists the latest documents", async () => {
    await page.click('a[data-view="dashboard"]');
    const t = await page.textContent("#dash-documents");
    assert(t.includes("Blood test March") && t.includes("Prescription Dr Rao (edited)"), t);
  });
  if (SHOTS) { await clearToasts(page); await page.screenshot({ path: path.join(SHOTS, "dashboard-1280-light.png"), fullPage: true }); }

  await step("voice: recording without a microphone shows a clear message", async () => {
    await page.click('a[data-view="voice"]');
    await page.click("#rec-start");
    await page.waitForFunction(() => document.querySelector("#rec-status").textContent.trim().length > 0);
    const msg = await page.textContent("#rec-status");
    assert(/microphone|record|upload/i.test(msg), msg);
    return msg.slice(0, 70);
  });

  await step("voice: save a note from an uploaded WAV, rename, play, attach to a document", async () => {
    await page.fill("#rec-title", "After the lab visit");
    await page.selectOption("#rec-doc", { label: "Blood test March" });
    await page.setInputFiles("#rec-file", path.join(tmp, "note.wav"));
    await page.click("#rec-file-save");
    await page.waitForSelector('.voice-card .voice-title:text("After the lab visit")');
    const card = page.locator(".voice-card").first();
    const text = await card.textContent();
    assert(text.includes("0:01") && text.includes("WAV") && text.includes("Blood test March"), text);
    await card.locator('[data-act="edit"]').click();
    await card.locator('input[name="title"]').fill("Questions after lab visit");
    await card.locator('button[type="submit"]').click();
    await page.waitForSelector('.voice-card .voice-title:text("Questions after lab visit")');
    await page.locator('.voice-card [data-act="play"]').first().click();
    await page.waitForSelector(".voice-card audio");
    const dur = await page.$eval(".voice-card audio", (a) => new Promise((r) => (a.readyState >= 1 ? r(a.duration) : (a.onloadedmetadata = () => r(a.duration)))));
    assert(dur > 1 && dur < 1.5, "audio duration " + dur);
  });
  if (SHOTS) { await clearToasts(page); await page.screenshot({ path: path.join(SHOTS, "voice-1280-light.png"), fullPage: true }); }

  await step("documents: 'Voice note' on a card preselects that document", async () => {
    await page.click('a[data-view="documents"]');
    await page.locator(".doc-card", { hasText: "Prescription Dr Rao" }).locator('[data-act="voice"]').click();
    await page.waitForSelector('[data-section="voice"]:not(.hidden)');
    const sel = await page.$eval("#rec-doc", (s) => s.options[s.selectedIndex].text);
    assert(sel.startsWith("Prescription Dr Rao"), sel);
  });

  /* ---------- scan view: mocked /api/scan ---------- */
  const pagePng = docPng.toString("base64");
  const MOCK_VERSION = "scanner-0.2-ft-real_only (mock)";
  const PHOTO_W = 900, PHOTO_H = 1200; // photo.png
  let mockNotFound = false;
  const formField = (body, name) => {
    const m = new RegExp(`name="${name}"\r\n\r\n([^\r]*)\r\n`).exec(body.toString("latin1"));
    return m ? m[1] : null;
  };
  const mockScanner = async (p) => {
    await p.route("**/api/scan/status", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ available: true, model_version: MOCK_VERSION, metrics: null }) }));
    await p.route("**/api/scan", (r) => {
      const body = r.request().postDataBuffer() || Buffer.alloc(0);
      const corners = formField(body, "corners");
      scanRequests.push({ mode: formField(body, "mode"), corners: corners ? JSON.parse(corners) : null, bytes: body.length, hasImage: body.includes('name="image"') });
      let res;
      if (corners) {
        const c = JSON.parse(corners);
        res = { page_found: true, corners: c, corners_norm: c.map(([x, y]) => [x / PHOTO_W, y / PHOTO_H]), confidence: null, method: "manual" };
      } else if (mockNotFound) {
        res = { page_found: false, corners: [[0, 0], [PHOTO_W, 0], [PHOTO_W, PHOTO_H], [0, PHOTO_H]], corners_norm: [[0, 0], [1, 0], [1, 1], [0, 1]], confidence: 0.12, method: "fallback" };
      } else {
        res = { page_found: true, corners: [[205, 115], [780, 185], [690, 1100], [120, 1020]],
          corners_norm: [[0.228, 0.096], [0.867, 0.154], [0.767, 0.917], [0.133, 0.85]], confidence: 0.93, method: "ml" };
      }
      return r.fulfill({ status: 200, contentType: "application/json",
        body: JSON.stringify({ ...res, width: 600, height: 800, image_png_b64: pagePng, model_version: MOCK_VERSION, elapsed_ms: 123.4 }) });
    });
  };
  const unmockScanner = async (p) => { await p.unroute("**/api/scan"); await p.unroute("**/api/scan/status"); };
  const scanIdle = (p, timeout = 10000) => p.waitForFunction(() => {
    const step = document.querySelector("#scan-step-check");
    return !step.hidden && !step.classList.contains("is-busy") && document.querySelector("#scan-page").getAttribute("src");
  }, null, { timeout });

  await step("scan: scanner missing on the server; the photo can still be saved without scanning", async () => {
    await page.route("**/api/scan/status", (r) => r.fulfill({ status: 404, contentType: "application/json", body: '{"error":"The scanner is not installed on this server."}' }));
    await page.click('a[data-view="scan"]');
    await page.waitForSelector('[data-section="scan"]:not(.hidden)');
    await page.waitForFunction(() => /not installed/.test(document.querySelector("#scan-avail").textContent));
    assert(/without scanning/.test(await page.textContent("#scan-avail")), "no 'without scanning' offer");
    assert(await page.isVisible("#scan-drop"), "photo step should stay usable");
    await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
    await page.waitForSelector("#scan-save:not([hidden])");
    assert((await page.getAttribute('#scan-steps li[data-step="check"]', "class")) === "skipped", "Check should be marked skipped");
    assert((await page.inputValue("#scan-title")).startsWith("Document photo"), "raw title");
    assert(/not scanned/.test(await page.textContent("#scan-save-cap")));
    assert(await page.evaluate(() => document.activeElement.id === "scan-h-save"), "focus should move to the Save heading");
    await page.click("#scan-discard");
    await page.waitForSelector("#scan-step-photo:not([hidden])");
    await page.unroute("**/api/scan/status");
  });

  await step("scan (mocked): Dashboard quick action, outline, confidence, mode switch", async () => {
    await mockScanner(page);
    await page.click('a[data-view="dashboard"]');
    await page.click("#dash-scan");
    await page.waitForSelector('[data-section="scan"]:not(.hidden)');
    assert(await page.evaluate(() => document.activeElement.id === "h-scan"), "focus on the Scan heading");
    assert((await page.getAttribute('a[data-view="scan"]', "aria-current")) === "page", "Scan nav item not current");
    await page.waitForFunction((v) => document.querySelector("#scan-model").textContent.includes(v), MOCK_VERSION);
    await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
    await scanIdle(page);
    assert((await page.getAttribute('#scan-steps li[data-step="check"]', "aria-current")) === "step", "step indicator");
    const pts = await page.getAttribute("#scan-outline polygon", "points");
    assert(pts && pts.split(" ").length === 4, "outline points " + pts);
    const status = await page.textContent("#scan-status");
    assert(/high confidence/.test(status) && status.includes("93%"), status);
    assert(await page.isHidden("#scan-notfound"));
    await page.check('input[name="scan-mode"][value="bw"]', { force: true });
    await scanIdle(page);
    assert(scanRequests.length === 2 && scanRequests[0].mode === "gray" && scanRequests[1].mode === "bw",
      "modes sent: " + JSON.stringify(scanRequests.map((s) => s.mode)));
    assert(scanRequests[0].hasImage && !scanRequests[0].corners, "first request: image, no corners");
    assert(/Black and white/.test(await page.textContent("#scan-meta")));
  });

  await step("scan (mocked): drag a corner, move one with the keyboard, Apply sends the corners field", async () => {
    await page.click("#scan-adjust");
    assert(await page.locator(".corner-handle").count() === 4, "four handles");
    assert(await page.evaluate(() => document.activeElement.classList.contains("corner-handle")), "focus on the first handle");
    const box = await page.locator("#scan-photo").boundingBox();
    const h0 = await page.locator('.corner-handle[data-i="0"]').boundingBox();
    await page.mouse.move(h0.x + h0.width / 2, h0.y + h0.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.15, box.y + box.height * 0.1, { steps: 6 });
    await page.mouse.move(box.x + box.width * 0.10, box.y + box.height * 0.05, { steps: 6 });
    await page.mouse.up();
    await page.focus('.corner-handle[data-i="1"]');
    await page.keyboard.press("Shift+ArrowRight");
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("ArrowDown");
    const label = await page.getAttribute('.corner-handle[data-i="1"]', "aria-label");
    assert(/Top-right corner, 92% across, 16% down/.test(label), label);
    await page.click("#scan-apply");
    await scanIdle(page);
    const last = scanRequests.at(-1);
    assert(last.corners && last.corners.length === 4 && last.mode === "bw", "corners not sent: " + JSON.stringify(last));
    const near = (a, b, tol) => Math.abs(a - b) <= tol;
    const [c0, c1] = last.corners;
    assert(near(c0[0], 0.10 * PHOTO_W, PHOTO_W * 0.02) && near(c0[1], 0.05 * PHOTO_H, PHOTO_H * 0.02), "dragged corner " + JSON.stringify(c0));
    assert(near(c1[0], 0.917 * PHOTO_W, 2) && near(c1[1], 0.164 * PHOTO_H, 2), "keyboard corner " + JSON.stringify(c1));
    assert(/your corners/.test(await page.textContent("#scan-status")));
    assert(await page.locator(".corner-handle").count() === 0, "handles should go after Apply");
    assert(await page.evaluate(() => document.activeElement.id === "scan-adjust"), "focus back on Adjust corners");
    // A mode switch now re-sends the user's corners.
    await page.check('input[name="scan-mode"][value="color"]', { force: true });
    await scanIdle(page);
    assert(scanRequests.at(-1).mode === "color" && scanRequests.at(-1).corners, "mode switch should keep the corners");
    return `${scanRequests.length} requests`;
  });

  await step("scan (mocked): save with metadata; success state; the scan appears in Documents", async () => {
    await page.click("#scan-next");
    await page.waitForSelector("#scan-save:not([hidden])");
    assert((await page.inputValue("#scan-title")).startsWith("Scanned document"), "title not prefilled");
    assert((await page.inputValue("#scan-date")).length === 10, "date not prefilled");
    await page.fill("#scan-title", "Scanned discharge summary");
    await page.selectOption("#scan-category", "discharge_summary");
    await page.fill("#scan-tags", "discharge, synthetic");
    await page.fill("#scan-notes", "Synthetic scan test");
    await page.click("#scan-save-btn");
    await page.waitForSelector("#scan-done:not([hidden])");
    assert((await page.textContent("#scan-done-text")).includes("Scanned discharge summary"));
    assert(await page.evaluate(() => document.activeElement.id === "scan-h-done"), "focus on the success heading");
    assert(await page.isVisible("#scan-another"), "Scan another");
    await page.click("#scan-view-docs");
    await page.waitForSelector('[data-section="documents"]:not(.hidden)');
    await page.waitForSelector('.doc-card .doc-title:text("Scanned discharge summary")');
    const card = page.locator(".doc-card", { hasText: "Scanned discharge summary" });
    const t = await card.textContent();
    assert(t.includes("PNG") && t.includes("Scanned") && t.includes("Discharge summary") && t.includes("synthetic"), t);
    const meta = await page.evaluate(() => Documents.list().find((d) => d.title === "Scanned discharge summary").scan);
    assert(meta && meta.method === "manual" && meta.mode === "color" && meta.modelVersion === "scanner-0.2-ft-real_only (mock)", JSON.stringify(meta));
    // The Documents button opens the scan view, back at step 1.
    assert(await page.locator("#doc-scan").count() === 0, "the old in-Documents scan panel should be gone");
    await page.click("#doc-scan-btn");
    await page.waitForSelector('[data-section="scan"]:not(.hidden)');
    assert(await page.isVisible("#scan-step-photo") && await page.isHidden("#scan-done"), "scan view should restart at step 1");
  });

  await step("scan (mocked): page edges not found explains, offers retake, corners and save anyway", async () => {
    mockNotFound = true;
    await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
    await scanIdle(page);
    const nf = await page.textContent("#scan-notfound");
    assert(/could not be found/.test(nf) && /whole photo/.test(nf) && /background/.test(nf), nf);
    assert((await page.getAttribute("#scan-outline polygon", "points")) === "", "no outline expected");
    assert(await page.isVisible("#scan-nf-retake") && await page.isVisible("#scan-nf-save"));
    await page.click("#scan-nf-adjust");
    assert(await page.locator(".corner-handle").count() === 4);
    await page.keyboard.press("Escape");
    assert(await page.locator(".corner-handle").count() === 0, "Escape should cancel corner editing");
    await page.click("#scan-nf-retake");
    await page.waitForSelector("#scan-step-photo:not([hidden])");
    assert(!(await page.getAttribute("#scan-photo", "src")), "photo should be cleared on retake");
    mockNotFound = false;
    await unmockScanner(page);
  });

  await step("scan with the real scanner (only when the server reports it available), including manual corners", async () => {
    const status = await page.evaluate(async () => {
      const r = await fetch("api/scan/status");
      return { code: r.status, body: r.ok ? await r.json() : null };
    });
    if (status.code !== 200) return `SKIPPED: /api/scan/status answered ${status.code} (scanner not installed)`;
    await page.evaluate(() => { location.hash = "#dashboard"; });
    await page.evaluate(() => { location.hash = "#scan"; });
    await page.waitForFunction(() => document.querySelector("#scan-model").textContent.length > 0);
    const model = await page.textContent("#scan-model");
    if (status.body.model_version) assert(model.includes(status.body.model_version), model);
    await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
    await scanIdle(page, 30000);
    const s = await page.textContent("#scan-status");
    await page.click("#scan-adjust");
    await page.click("#scan-apply");
    await scanIdle(page, 30000);
    const s2 = await page.textContent("#scan-status");
    assert(/your corners/.test(s2), "real manual corners: " + s2);
    await page.click("#scan-retake");
    return `real scanner, available=${status.body.available}: ${s.slice(0, 70)} | manual: ok`;
  });

  await step("storage used is shown per user", async () => {
    await page.click('a[data-view="profile"]');
    await page.waitForFunction(() => /3 files/.test(document.querySelector("#storage-facts").textContent));
    const t = await page.textContent("#storage-facts");
    assert(/1 note/.test(t), t);
    assert(/Files: .*3 documents, 1 voice note/.test(await page.textContent(".side-storage")));
  });
  if (SHOTS) { await clearToasts(page); await page.screenshot({ path: path.join(SHOTS, "profile-1280-light.png"), fullPage: true }); }

  await step("deleting a document keeps its voice note and unlinks it", async () => {
    await page.click('a[data-view="documents"]');
    await page.locator(".doc-card", { hasText: "Blood test March" }).locator('[data-act="delete"]').click();
    await page.waitForFunction(() => document.querySelectorAll(".doc-card").length === 2);
    const notes = await page.evaluate(async () => FilesDB.listVoiceNotes(App.user().id));
    assert(notes.length === 1 && notes[0].documentId === "", JSON.stringify(notes));
  });

  /* ---------- reading text (OCR): mocked /api/ocr ---------- */
  const OCR_TEXT = "SYNTHETIC LAB REPORT\nHaemoglobin 13.5 g/dL";
  let ocrMode = "ok"; // ok | low | empty | error
  let ocrRequests = [];
  let ocrStatus = { available: true, models: { detector: "pretrained", recogniser: "finetuned" } };
  const mockOcr = async (p) => {
    await p.route("**/api/ocr/status", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(ocrStatus) }));
    await p.route("**/api/ocr", async (r) => {
      const body = r.request().postDataBuffer() || Buffer.alloc(0);
      ocrRequests.push({ hasImage: body.includes('name="image"'), scanner: formField(body, "scanner"), bytes: body.length });
      await new Promise((res) => setTimeout(res, 250)); // long enough to see the progress state
      if (ocrMode === "error") {
        return r.fulfill({ status: 503, contentType: "application/json", body: '{"error":"The text-reading models could not be loaded (mock)."}' });
      }
      const text = ocrMode === "empty" ? "" : OCR_TEXT;
      const conf = ocrMode === "low" ? 0.41 : 0.93;
      const words = text.split(/\s+/).filter(Boolean).map((t, i) => ({ text: t, confidence: conf, box: [10 + i * 50, 10, 50 + i * 50, 30], line: i < 3 ? 0 : 1 }));
      const lines = text ? text.split("\n").map((t, i) => ({ text: t, box: [10, 10 + i * 30, 400, 30 + i * 30], confidence: conf })) : [];
      return r.fulfill({ status: 200, contentType: "application/json",
        body: JSON.stringify({ text, lines, words, page: { width: 600, height: 800 }, seconds: 0.25, models: ocrStatus.models }) });
    });
  };
  const unmockOcr = async (p) => { await p.unroute("**/api/ocr"); await p.unroute("**/api/ocr/status"); };
  const readEnabled = (p, sel) => p.waitForFunction((s) => { const b = document.querySelector(s); return b && !b.hidden && !b.disabled; }, sel);
  const textShown = (p, root) => p.waitForFunction((r) => {
    const box = document.querySelector(r);
    return box && !box.querySelector(".ocr-panel").hidden && box.querySelector(".ocr-busy").hidden;
  }, root);

  await step("read text (mocked): the action appears after a scan, shows progress, then the text with a word count and Copy", async () => {
    await ctx.grantPermissions(["clipboard-read", "clipboard-write"]);
    await mockScanner(page);
    await mockOcr(page);
    await page.click('a[data-view="scan"]');
    await page.waitForSelector("#scan-step-photo:not([hidden])");
    assert(await page.isHidden("#scan-ocr"), "no text box before a page is scanned");
    await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
    await scanIdle(page);
    await readEnabled(page, "#scan-ocr-read");
    assert((await page.textContent("#scan-ocr-read")).trim() === "Read text", "button label");
    assert(await page.isHidden("#scan-ocr .ocr-panel"), "no text panel before reading");
    assert(/copy it and find this document/.test(await page.textContent("#scan-ocr-hint")), "idle hint");
    await page.click("#scan-ocr-read");
    await page.waitForSelector("#scan-ocr .ocr-busy:not([hidden])");
    assert(/Reading text/.test(await page.textContent("#scan-ocr .ocr-busy")), "progress text");
    assert((await page.getAttribute('#scan-ocr [data-ocr="live"]', "aria-live")) === "polite", "progress is announced");
    assert(await page.isDisabled("#scan-ocr-read") && await page.isDisabled("#scan-next"), "Read text and Continue wait while reading");
    await textShown(page, "#scan-ocr");
    assert((await page.inputValue("#scan-ocr-text")) === OCR_TEXT, "text: " + await page.inputValue("#scan-ocr-text"));
    assert((await page.textContent("#scan-ocr-count")) === "6 words", await page.textContent("#scan-ocr-count"));
    assert(await page.isHidden("#scan-ocr-note"), "no note for a confident read");
    assert(/6 words/.test(await page.textContent('#scan-ocr [data-ocr="live"]')), "result announced");
    assert(await page.isEnabled("#scan-next"), "Continue enabled again");
    assert((await page.textContent("#scan-ocr-read")).trim() === "Read again");
    assert(ocrRequests.length === 1 && ocrRequests[0].hasImage && ocrRequests[0].scanner === null, JSON.stringify(ocrRequests));
    assert(await page.evaluate(() => document.querySelector('label[for="scan-ocr-text"]') !== null), "textarea has a label");
    await page.click("#scan-ocr-copy");
    await page.waitForFunction(() => /Copied/.test(document.querySelector('#scan-ocr [data-ocr="done"]').textContent));
    assert((await page.evaluate(() => navigator.clipboard.readText())) === OCR_TEXT, "clipboard content");
  });

  await step("read text (mocked): low-confidence note, no-text note, inline error, and a changed page is flagged", async () => {
    ocrMode = "low";
    await page.click("#scan-ocr-read");
    await page.waitForFunction(() => !document.querySelector("#scan-ocr-note").hidden);
    const low = await page.textContent("#scan-ocr-note");
    assert(/hard to read/.test(low) && low.includes("41%"), low);
    ocrMode = "empty";
    await page.click("#scan-ocr-read");
    await page.waitForFunction(() => /No text was found/.test(document.querySelector("#scan-ocr-note").textContent));
    assert((await page.textContent("#scan-ocr-count")) === "0 words" && await page.isDisabled("#scan-ocr-copy"), "empty text: nothing to copy");
    assert(await page.isVisible("#scan-ocr-text"), "the textarea stays so the text can be typed");
    ocrMode = "error";
    await page.click("#scan-ocr-read");
    await page.waitForSelector("#scan-ocr-error:not([hidden])");
    const err = await page.textContent("#scan-ocr-error");
    assert(/could not be read/.test(err) && /could not be loaded \(mock\)/.test(err), err);
    assert(await page.isVisible("#scan-page") && await page.isEnabled("#scan-ocr-read") && await page.isEnabled("#scan-next"), "the step stays usable after an error");
    ocrMode = "ok";
    await page.click("#scan-ocr-read");
    await page.waitForFunction((t) => document.querySelector("#scan-ocr-text").value === t, OCR_TEXT);
    assert(await page.isHidden("#scan-ocr-error"), "error cleared by a good read");
    await page.check('input[name="scan-mode"][value="bw"]', { force: true });
    await scanIdle(page);
    assert(/page has changed/.test(await page.textContent("#scan-ocr-hint")), "stale hint after a mode switch");
    assert((await page.inputValue("#scan-ocr-text")) === OCR_TEXT, "text kept after a mode switch");
  });

  const EDITED = OCR_TEXT + "\nChecked by hand: ferritin zebrafinch";
  await step("read text: the edited text is saved with the scan, flagged and searchable in Documents, shown in the viewer", async () => {
    await page.fill("#scan-ocr-text", EDITED);
    assert((await page.textContent("#scan-ocr-count")) === "11 words", await page.textContent("#scan-ocr-count"));
    await page.click("#scan-next");
    await page.waitForSelector("#scan-save:not([hidden])");
    assert(/11 words/.test(await page.textContent("#scan-save-text")), "the save step says the text is saved too");
    await page.fill("#scan-title", "Scan with text");
    await page.click("#scan-save-btn");
    await page.waitForSelector("#scan-done:not([hidden])");
    await page.click("#scan-view-docs");
    await page.waitForSelector('.doc-card .doc-title:text("Scan with text")');
    const card = page.locator(".doc-card", { hasText: "Scan with text" });
    assert(await card.locator(".text-chip").count() === 1, "Text indicator on the card");
    const other = page.locator(".doc-card", { hasText: "Prescription Dr Rao" });
    assert(await other.locator(".text-chip").count() === 0, "no indicator on a document without text");
    const rec = await page.evaluate(() => Documents.list().find((d) => d.title === "Scan with text"));
    assert(rec.text === EDITED, "stored text: " + rec.text);
    assert(rec.ocr && rec.ocr.edited === true && rec.ocr.words === 11 && Math.abs(rec.ocr.confidence - 0.93) < 0.001, JSON.stringify(rec.ocr));
    await page.fill("#doc-q", "zebrafinch");
    await page.waitForFunction(() => document.querySelectorAll(".doc-card").length === 1);
    assert((await page.textContent(".doc-card .doc-title")) === "Scan with text", "search by saved text");
    await page.fill("#doc-q", "");
    await page.waitForFunction(() => document.querySelectorAll(".doc-card").length > 1);
    await card.locator('.card-actions [data-act="view"]').click();
    await page.waitForSelector("#viewer[open] #viewer-ocr-text");
    assert((await page.inputValue("#viewer-ocr-text")) === EDITED, "viewer text");
    assert(await page.isDisabled("#viewer-ocr-save"), "nothing to save until the text is changed");
    await page.click("#viewer-ocr-copy");
    await page.waitForFunction(() => /Copied/.test(document.querySelector('#viewer-extra [data-ocr="done"]').textContent));
    assert((await page.evaluate(() => navigator.clipboard.readText())) === EDITED, "clipboard content from the viewer");
    await page.fill("#viewer-ocr-text", EDITED + " ok");
    await page.click("#viewer-ocr-save");
    await page.waitForFunction(() => /Saved/.test(document.querySelector('#viewer-extra [data-ocr="done"]').textContent));
    const again = await page.evaluate(() => Documents.list().find((d) => d.title === "Scan with text").text);
    assert(again === EDITED + " ok", "text corrected in the viewer: " + again);
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, "viewer-text-1280-light.png") });
    await page.keyboard.press("Escape");
    await page.waitForSelector("#viewer:not([open])", { state: "attached" });
  });

  await step("read text: a document stored before text existed still loads; a PDF offers no Read text; an uploaded image does", async () => {
    // Put the PDF's record back into the shape written before this feature (no text, no ocr fields).
    await page.evaluate(async () => {
      const d = Documents.list().find((x) => x.mime === "application/pdf");
      const db = await FilesDB.open();
      await new Promise((res, rej) => {
        const tx = db.transaction("documents", "readwrite");
        const old = { ...d };
        delete old.text; delete old.ocr;
        tx.objectStore("documents").put(old);
        tx.oncomplete = res; tx.onerror = () => rej(tx.error);
      });
      await Documents.refresh();
    });
    const legacy = await page.evaluate(() => Documents.list().find((x) => x.mime === "application/pdf"));
    assert(!("text" in legacy) && !("ocr" in legacy), "legacy record shape");
    const pdf = page.locator(".doc-card", { hasText: "Prescription Dr Rao" });
    assert(await pdf.count() === 1 && await pdf.locator(".text-chip").count() === 0, "legacy document listed without the indicator");
    await page.fill("#doc-q", "rao");
    await page.waitForFunction(() => document.querySelectorAll(".doc-card").length === 1);
    await page.fill("#doc-q", "");
    await page.waitForFunction(() => document.querySelectorAll(".doc-card").length > 1);
    await pdf.locator('.card-actions [data-act="view"]').click();
    await page.waitForSelector("#viewer[open] #viewer-body iframe");
    assert(await page.isHidden("#viewer-extra") && await page.locator("#viewer-ocr-read").count() === 0, "no Read text for a PDF");
    await page.keyboard.press("Escape");
    await page.waitForSelector("#viewer:not([open])", { state: "attached" });
    // An uploaded image: Read text in the viewer, saved at once.
    await page.click("#doc-add-btn");
    await page.setInputFiles("#doc-file", path.join(tmp, "lab-report.png"));
    await page.fill("#up-title", "Uploaded image");
    await page.click("#doc-upload-save");
    await page.waitForSelector('.doc-card .doc-title:text("Uploaded image")');
    const img = page.locator(".doc-card", { hasText: "Uploaded image" });
    assert(await img.locator(".text-chip").count() === 0, "no indicator before reading");
    await img.locator('.card-actions [data-act="view"]').click();
    await readEnabled(page, "#viewer-ocr-read");
    assert(/No text has been read/.test(await page.textContent("#viewer-ocr-hint")), "hint in the viewer");
    const before = ocrRequests.length;
    await page.click("#viewer-ocr-read");
    await textShown(page, "#viewer-extra");
    await page.waitForFunction(() => /Saved with the document/.test(document.querySelector('#viewer-extra [data-ocr="done"]').textContent));
    assert(ocrRequests.length === before + 1 && ocrRequests.at(-1).hasImage, "the stored image was sent");
    assert((await page.inputValue("#viewer-ocr-text")) === OCR_TEXT);
    await page.keyboard.press("Escape");
    await page.waitForSelector("#viewer:not([open])", { state: "attached" });
    const stored = await page.evaluate(() => Documents.list().find((d) => d.title === "Uploaded image"));
    assert(stored.text === OCR_TEXT && stored.ocr.edited === false, JSON.stringify(stored.ocr));
    assert(await page.locator(".doc-card", { hasText: "Uploaded image" }).locator(".text-chip").count() === 1, "indicator after reading");
  });

  await step("read text: when the server reports it unavailable the actions are disabled with a one-line reason", async () => {
    ocrStatus = { available: false, reason: "Reading text needs extra packages on the server (pip install -r requirements-ocr.txt)." };
    const before = ocrRequests.length;
    await page.click('a[data-view="scan"]');
    await page.waitForSelector("#scan-step-photo:not([hidden])");
    await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
    await scanIdle(page);
    await page.waitForFunction(() => /not available/.test(document.querySelector("#scan-ocr-hint").textContent));
    assert(await page.isDisabled("#scan-ocr-read"), "Read text disabled in the scan view");
    assert((await page.textContent("#scan-ocr-hint")).includes("requirements-ocr.txt"), "reason shown");
    assert((await page.getAttribute("#scan-ocr-read", "aria-describedby")) === "scan-ocr-hint", "the reason describes the button");
    await page.click("#scan-ocr-read", { force: true });
    assert(await page.isHidden("#scan-ocr-error") && ocrRequests.length === before, "a click does nothing and fails nothing");
    assert(await page.isEnabled("#scan-next"), "the scan can still be saved");
    await page.click("#scan-retake");
    // Viewer: saved text is still shown and can be copied; reading again is disabled.
    await page.click('a[data-view="documents"]');
    await page.locator(".doc-card", { hasText: "Uploaded image" }).locator('.card-actions [data-act="view"]').click();
    await page.waitForFunction(() => /not available/.test((document.querySelector("#viewer-ocr-hint") || {}).textContent || ""));
    assert(await page.isDisabled("#viewer-ocr-read") && await page.isEnabled("#viewer-ocr-copy"), "viewer: read disabled, copy enabled");
    assert((await page.inputValue("#viewer-ocr-text")) === OCR_TEXT);
    await page.keyboard.press("Escape");
    await page.waitForSelector("#viewer:not([open])", { state: "attached" });
    ocrStatus = { available: true, models: { detector: "pretrained", recogniser: "finetuned" } };
    await unmockOcr(page);
    await unmockScanner(page);
  });

  await step("read text with the real OCR (only when the server reports it available)", async () => {
    const status = await page.evaluate(async () => {
      const r = await fetch("api/ocr/status");
      return { code: r.status, body: r.ok ? await r.json() : null };
    });
    if (status.code !== 200 || !status.body.available) {
      return `SKIPPED: ${status.code === 200 ? status.body.reason : "/api/ocr/status answered " + status.code}`;
    }
    await mockScanner(page); // the mocked page image carries the words SYNTHETIC LAB REPORT
    await page.click('a[data-view="dashboard"]');
    await page.click('a[data-view="scan"]');
    await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
    await scanIdle(page);
    await readEnabled(page, "#scan-ocr-read");
    await page.click("#scan-ocr-read");
    await page.waitForFunction(() => {
      const box = document.querySelector("#scan-ocr");
      return !box.querySelector(".ocr-panel").hidden || !box.querySelector(".ocr-error").hidden;
    }, null, { timeout: 120000 });
    assert(await page.isHidden("#scan-ocr-error"), "real OCR error: " + await page.textContent("#scan-ocr-error"));
    const text = await page.inputValue("#scan-ocr-text");
    assert(/SYNTHETIC/i.test(text) && /REPORT/i.test(text), "real OCR text: " + text);
    await page.click("#scan-retake");
    await unmockScanner(page);
    return `real OCR read: ${text.replace(/\s+/g, " ").slice(0, 40)}`;
  });

  /* ---------- responsive and dark ---------- */
  const views = ["dashboard", "vitals", "conditions", "medications", "allergies", "visits", "documents", "scan", "voice", "profile"];
  await step("no horizontal scroll at 360 px on any view; mobile menu works with the keyboard", async () => {
    await page.setViewportSize({ width: 360, height: 740 });
    const wide = [];
    for (const v of views) {
      await page.evaluate((v) => { location.hash = "#" + v; }, v);
      await page.waitForSelector(`[data-section="${v}"]:not(.hidden)`);
      const sw = await page.evaluate(() => document.documentElement.scrollWidth);
      if (sw > 360) wide.push(`${v}: ${sw}px`);
    }
    // The scan check step with the corner handles out, and the save step.
    await mockScanner(page);
    await page.evaluate(() => { location.hash = "#scan"; });
    await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
    await scanIdle(page);
    await page.click("#scan-adjust");
    let sw = await page.evaluate(() => document.documentElement.scrollWidth);
    if (sw > 360) wide.push(`scan (adjust corners): ${sw}px`);
    await page.click("#scan-adjust-cancel");
    await mockOcr(page);
    await page.evaluate(() => Scan.checkStatus()); // nothing changes; gives the OCR status mock time to apply
    await page.evaluate(() => { location.hash = "#dashboard"; });
    await page.evaluate(() => { location.hash = "#scan"; });
    await readEnabled(page, "#scan-ocr-read");
    await page.click("#scan-ocr-read");
    await textShown(page, "#scan-ocr");
    sw = await page.evaluate(() => document.documentElement.scrollWidth);
    if (sw > 360) wide.push(`scan (text panel): ${sw}px`);
    await page.click("#scan-next");
    sw = await page.evaluate(() => document.documentElement.scrollWidth);
    if (sw > 360) wide.push(`scan (save): ${sw}px`);
    await page.click("#scan-save-btn");
    await page.waitForSelector("#scan-done:not([hidden])");
    await page.click("#scan-view-docs");
    await page.locator(".doc-card .card-actions [data-act=\"view\"]").first().click();
    await page.waitForSelector("#viewer[open] #viewer-ocr-text");
    const dlg = await page.evaluate(() => { const d = document.querySelector("#viewer"); return [d.scrollWidth, d.clientWidth]; });
    if (dlg[0] > dlg[1]) wide.push(`viewer with text: ${dlg[0]} > ${dlg[1]}px`);
    await page.keyboard.press("Escape");
    await page.waitForSelector("#viewer:not([open])", { state: "attached" });
    await unmockOcr(page);
    await unmockScanner(page);
    assert(!wide.length, "too wide: " + wide.join(", "));
    assert(await page.isHidden("#sidebar"), "sidebar should be collapsed on mobile");
    await page.focus("#menu-btn");
    await page.keyboard.press("Enter");
    await page.waitForSelector("#sidebar", { state: "visible" });
    assert((await page.getAttribute("#menu-btn", "aria-expanded")) === "true");
    await page.keyboard.press("Escape");
    await page.waitForSelector("#sidebar", { state: "hidden" });
    await page.click("#menu-btn");
    await page.click('a[data-view="vitals"]');
    await page.waitForSelector('[data-section="vitals"]:not(.hidden)');
    assert(await page.isHidden("#sidebar"), "menu should close after choosing a view");
  });

  if (SHOTS) {
    for (const scheme of ["light", "dark"]) {
      await page.emulateMedia({ colorScheme: scheme });
      for (const [w, h] of [[1280, 900], [375, 812]]) {
        await page.setViewportSize({ width: w, height: h });
        for (const v of ["dashboard", "vitals", "documents", "voice"]) {
          await page.evaluate((v) => { location.hash = "#" + v; }, v);
          await page.waitForSelector(`[data-section="${v}"]:not(.hidden)`);
          await page.waitForTimeout(50);
          await clearToasts(page); await page.screenshot({ path: path.join(SHOTS, `${v}-${w}-${scheme}.png`), fullPage: true });
        }
      }
    }
    await page.emulateMedia({ colorScheme: "light" });
    await page.setViewportSize({ width: 1280, height: 900 });
  }

  if (SHOTS) {
    // The scan flow, step by step (mocked scanner), at 1280 and 375 px, light and dark.
    await mockScanner(page);
    for (const scheme of ["light", "dark"]) {
      await page.emulateMedia({ colorScheme: scheme });
      for (const [w, h] of [[1280, 900], [375, 812]]) {
        await page.setViewportSize({ width: w, height: h });
        const shot = async (name) => {
          await clearToasts(page);
          await page.evaluate(() => window.scrollTo(0, 0));
          await page.waitForTimeout(80);
          await page.screenshot({ path: path.join(SHOTS, `scan-${name}-${w}-${scheme}.png`), fullPage: true });
        };
        await page.evaluate(() => { location.hash = "#dashboard"; });
        await page.evaluate(() => { location.hash = "#scan"; });
        await page.waitForSelector("#scan-step-photo:not([hidden])");
        await page.waitForFunction(() => document.querySelector("#scan-model").textContent.length > 0);
        await shot("1-photo");
        await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
        await scanIdle(page);
        await shot("2-check");
        await page.click("#scan-adjust");
        await shot("2-adjust");
        await page.click("#scan-adjust-cancel");
        await page.click("#scan-next");
        await shot("3-save");
        await page.click("#scan-save-btn");
        await page.waitForSelector("#scan-done:not([hidden])");
        await shot("4-done");
        await page.click("#scan-another");
        mockNotFound = true;
        await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
        await scanIdle(page);
        await shot("2-notfound");
        mockNotFound = false;
        await page.click("#scan-retake");
      }
    }
    await unmockScanner(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.setViewportSize({ width: 1280, height: 900 });
  }

  await step("deleting the user deletes their documents and voice notes", async () => {
    await page.setViewportSize({ width: 1280, height: 900 });
    const id = await page.evaluate(() => App.user().id);
    await page.evaluate(() => { location.hash = "#profile"; });
    await page.click("#delete-user");
    await page.waitForSelector("#auth-screen:not(.hidden)");
    const left = await page.evaluate(async (id) => FilesDB.usage(id), id);
    assert(left.documents.count === 0 && left.voiceNotes.count === 0, JSON.stringify(left));
    assert(!(await page.textContent("#user-list")).includes("Asha Test"));
  });

  await step("recording with a (fake) microphone: record, stop, preview, save", async () => {
    const b2 = await chromium.launch({ args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] });
    try {
      const c2 = await b2.newContext({ permissions: ["microphone"] });
      const p2 = await c2.newPage();
      p2.on("pageerror", (e) => pageErrors.push(`rec: ${e.message}`));
      await p2.goto(BASE + "/");
      await p2.fill('#new-user-form [name="name"]', "Mic Test");
      await p2.fill('#new-user-form [name="dob"]', "2000-01-01");
      await p2.click('#new-user-form button[type="submit"]');
      await p2.evaluate(() => { location.hash = "#voice"; });
      await p2.click("#rec-start");
      await p2.waitForSelector("#rec-stop:not([disabled])");
      await p2.waitForTimeout(1300);
      await p2.click("#rec-stop");
      await p2.waitForSelector("#rec-preview:not([hidden])");
      await p2.click("#rec-save");
      await p2.waitForSelector(".voice-card");
      const t = await p2.textContent(".voice-card");
      assert(/0:01|0:02/.test(t), t);
      return t.replace(/\s+/g, " ").trim().slice(0, 60);
    } finally {
      await b2.close();
    }
  });

  await step("opened as file://: scanning explains it needs python server.py", async () => {
    const { ctx: c3, page: p3 } = await newPage();
    await p3.goto("file://" + path.join(ROOT, "index.html"));
    await p3.fill('#new-user-form [name="name"]', "File Mode");
    await p3.fill('#new-user-form [name="dob"]', "2000-01-01");
    await p3.click('#new-user-form button[type="submit"]');
    await p3.evaluate(() => { location.hash = "#documents"; });
    await p3.click("#doc-scan-btn");
    await p3.waitForSelector('[data-section="scan"]:not(.hidden)');
    const msg = await p3.textContent("#scan-avail");
    assert(msg.includes("python server.py") && msg.includes("without scanning"), msg);
    await p3.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
    await p3.waitForSelector("#scan-save:not([hidden])");
    await p3.fill("#scan-title", "Photo saved offline");
    await p3.click("#scan-save-btn");
    await p3.waitForSelector("#scan-done:not([hidden])");
    await p3.click("#scan-view-docs");
    await p3.waitForSelector('.doc-card .doc-title:text("Photo saved offline")');
    const offline = await p3.locator(".doc-card", { hasText: "Photo saved offline" }).textContent();
    assert(offline.includes("PNG") && !offline.includes("Scanned"), offline);
    // Documents still work without the server.
    await p3.click("#doc-add-btn");
    await p3.setInputFiles("#doc-file", path.join(tmp, "prescription.pdf"));
    await p3.click("#doc-upload-save");
    await p3.waitForSelector(".doc-card");
    await c3.close();
  });

  await step("no JavaScript errors on any page", async () => {
    assert(!pageErrors.length, pageErrors.join("\n"));
  });

  await ctx.close();
  await browser.close();
  if (server) server.kill();
  fs.rmSync(tmp, { recursive: true, force: true });

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed.`);
  process.exit(failed.length ? 1 : 0);
}

main().catch((err) => { console.error(err); process.exit(2); });

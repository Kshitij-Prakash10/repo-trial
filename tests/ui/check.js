#!/usr/bin/env node
/*
 * Health Diary UI check: drives the main flows in headless Chromium with Playwright.
 *
 *   node tests/ui/check.js
 *
 * Starts `python server.py` on a free local port (or uses BASE_URL if set), then checks: an older saved
 * user still opens, create user, add vitals, add / search / edit / view documents (PNG, PDF, a rejected
 * fake file), voice note from an uploaded WAV (rename, play, attach to a document), recording with a fake
 * microphone, the scan flow against a mocked /api/scan (and the real scanner when the server reports it
 * available), scanner-missing and file:// messages, storage shown, deleting a user deletes their files,
 * no horizontal scroll at 360 px, keyboard menu on mobile, no page errors.
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

  await step("scan: scanner missing on the server shows a clear message", async () => {
    await page.route("**/api/scan/status", (r) => r.fulfill({ status: 404, contentType: "application/json", body: '{"error":"The scanner is not installed on this server."}' }));
    await page.click('a[data-view="documents"]');
    await page.click("#doc-scan-btn");
    await page.waitForFunction(() => /not installed/.test(document.querySelector("#scan-avail").textContent));
    assert(await page.isHidden("#scan-body"));
    await page.click("#scan-close");
    await page.unroute("**/api/scan/status");
  });

  await step("scan (mocked /api/scan): outline, corrected page, mode switch, save as PNG document", async () => {
    const pagePng = docPng.toString("base64");
    await page.route("**/api/scan/status", (r) => r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ available: true, model_version: "mock-1", metrics: null }) }));
    await page.route("**/api/scan", (r) => {
      const body = r.request().postDataBuffer() || Buffer.alloc(0);
      const m = /name="mode"\r\n\r\n(\w+)/.exec(body.toString("latin1"));
      scanRequests.push({ mode: m && m[1], bytes: body.length, hasImage: body.includes('name="image"') });
      return r.fulfill({
        status: 200, contentType: "application/json",
        body: JSON.stringify({
          page_found: true, corners: [[205, 115], [780, 185], [690, 1100], [120, 1020]],
          corners_norm: [[0.228, 0.096], [0.867, 0.154], [0.767, 0.917], [0.133, 0.85]],
          confidence: 0.93, method: "ml", width: 600, height: 800, image_png_b64: pagePng,
          model_version: "mock-1", elapsed_ms: 123.4,
        }),
      });
    });
    await page.click("#doc-scan-btn");
    await page.waitForFunction(() => /Scanner ready/.test(document.querySelector("#scan-avail").textContent));
    await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
    await page.waitForSelector("#scan-result:not([hidden])");
    const pts = await page.getAttribute("#scan-outline polygon", "points");
    assert(pts && pts.split(" ").length === 4, "outline points " + pts);
    assert(await page.locator(".scan-photo-wrap .corner").count() === 4, "corner markers");
    assert((await page.textContent("#scan-status")).includes("93%"));
    await page.check('input[name="scan-mode"][value="bw"]', { force: true });
    await page.waitForFunction(() => !document.querySelector("#scan-mode-set").hasAttribute("aria-busy"));
    await page.waitForTimeout(100);
    assert(scanRequests.length >= 2 && scanRequests[0].mode === "gray" && scanRequests.at(-1).mode === "bw",
      "modes sent: " + JSON.stringify(scanRequests.map((s) => s.mode)));
    assert(scanRequests[0].hasImage, "image field not sent");
    if (SHOTS) { await clearToasts(page); await page.screenshot({ path: path.join(SHOTS, "scan-1280-light.png"), fullPage: true }); }
    await page.fill("#scan-title", "Scanned discharge summary");
    await page.selectOption("#scan-category", "discharge_summary");
    await page.click("#scan-save-btn");
    await page.waitForSelector('.doc-card .doc-title:text("Scanned discharge summary")');
    const card = page.locator(".doc-card", { hasText: "Scanned discharge summary" });
    assert((await card.textContent()).includes("PNG") && (await card.textContent()).includes("Scanned"));
    assert(await page.isHidden("#doc-scan"), "scanner panel should close after saving");
    await page.unroute("**/api/scan");
    await page.unroute("**/api/scan/status");
    return `${scanRequests.length} requests`;
  });

  await step("scan with the real scanner (only when the server reports it available)", async () => {
    const status = await page.evaluate(async () => {
      const r = await fetch("api/scan/status");
      return { code: r.status, body: r.ok ? await r.json() : null };
    });
    if (status.code !== 200) return `SKIPPED: /api/scan/status answered ${status.code} (scanner not installed)`;
    await page.click("#doc-scan-btn");
    await page.setInputFiles("#scan-file", path.join(tmp, "photo.png"));
    await page.waitForSelector("#scan-result:not([hidden])", { timeout: 30000 });
    const s = await page.textContent("#scan-status");
    await page.click("#scan-close");
    return `real scanner, available=${status.body.available}: ${s.slice(0, 80)}`;
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

  /* ---------- responsive and dark ---------- */
  const views = ["dashboard", "vitals", "conditions", "medications", "allergies", "visits", "documents", "voice", "profile"];
  await step("no horizontal scroll at 360 px on any view; mobile menu works with the keyboard", async () => {
    await page.setViewportSize({ width: 360, height: 740 });
    const wide = [];
    for (const v of views) {
      await page.evaluate((v) => { location.hash = "#" + v; }, v);
      await page.waitForSelector(`[data-section="${v}"]:not(.hidden)`);
      const sw = await page.evaluate(() => document.documentElement.scrollWidth);
      if (sw > 360) wide.push(`${v}: ${sw}px`);
    }
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
    const msg = await p3.textContent("#scan-avail");
    assert(msg.includes("python server.py"), msg);
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

/*
 * Health Diary: reading the text on a page (OCR) with the local server.
 *
 *   GET  api/ocr/status  -> { available, reason?, pipeline: "v2" | "v1", capabilities: { handwriting, orientation, deskew } }
 *   POST api/ocr (image, deskew?) -> { text, words: [{ text, confidence, ... }],
 *                                      lines: [{ text, confidence, confirm_numbers, script, ... }], preprocessing, ... }
 *
 * Ocr.widget(root, options) builds the "Text" box used in the Scan view and in the document viewer:
 * a Read text button, a progress line, and after reading an editable textarea with a word count, a Copy
 * button and a note when the text was hard to read. When the server cannot read text the button is
 * disabled and a one-line hint says why. Uses UI. Nothing leaves this computer.
 *
 * With the "v2" models on the server, handwriting is read too and turned pages are put upright. A
 * line with a number or a dose unit comes back with confirm_numbers: such numbers are
 * misread with high confidence (2.5 mg read as 25 mg), so the box always shows a warning listing those
 * lines, whatever the mean confidence. The warning is stored with the text (ocr.confirmNumbers). Once the
 * user has corrected the text by hand it turns into a shorter reminder; it goes away only when the page
 * is read again without such lines.
 */
const Ocr = (() => {
  const { $, esc, icon, fmtBytes } = UI;
  const PRESCRIPTION_CHECK = "Compare every medicine name, dose and unit with the prescription image. Check for any missing lines.";
  const MAX_BYTES = 10 * 1024 * 1024; // bytes, the server's limit for one image
  const MAX_CHARS = 200000; // characters kept with a document
  const LOW_CONFIDENCE = 0.6; // mean word confidence (0..1) below which the text gets a "check it" note
  const CONFIRM_SHOWN = 3; // flagged lines quoted in the warning; the rest are counted
  const CONFIRM_KEPT = 20; // flagged lines stored with the document
  const CONFIRM_LINE_CHARS = 120; // characters kept of each flagged line
  const NO_CAPS = { handwriting: false, orientation: false, deskew: false };
  const SERVER_HINT = "Start the local server with `python server.py` and open http://127.0.0.1:8000.";

  /**
   * Whether the server can read text: { available: bool, reason: string, capabilities: { handwriting,
   * orientation, deskew } }. Older servers send no capabilities: everything false. Never rejects.
   */
  async function status() {
    if (location.protocol === "file:") {
      return { available: false, reason: `It needs the local server: this page was opened as a file. ${SERVER_HINT}`, capabilities: NO_CAPS };
    }
    let r;
    try {
      r = await fetch("api/ocr/status", { headers: { Accept: "application/json" }, cache: "no-store" });
    } catch (err) {
      return { available: false, reason: `The local server did not answer. ${SERVER_HINT}`, capabilities: NO_CAPS };
    }
    const s = r.ok ? await r.json().catch(() => null) : null;
    const c = (s && s.capabilities) || {};
    const capabilities = { handwriting: c.handwriting === true, orientation: c.orientation === true, deskew: c.deskew === true };
    if (s && s.available) return { available: true, reason: "", capabilities };
    if (s) return { available: false, reason: s.reason || "The server cannot read text.", capabilities };
    return { available: false, reason: r.status === 404
      ? "It is not installed on this server." : `The server answered ${r.status}.`, capabilities: NO_CAPS };
  }

  /** The texts of the lines the server flagged with confirm_numbers (at most CONFIRM_KEPT, each shortened). */
  function flaggedLines(lines) {
    return (Array.isArray(lines) ? lines : [])
      .filter((ln) => ln && ln.confirm_numbers === true)
      .map((ln) => String(ln.text || "").trim().slice(0, CONFIRM_LINE_CHARS));
  }

  /** { count, lines } from a stored ocr info, or null when nothing was flagged (also for older records). */
  function confirmOf(info) {
    const c = info && info.confirmNumbers;
    if (!c || typeof c !== "object") return null;
    const lines = Array.isArray(c.lines) ? c.lines.filter((t) => typeof t === "string") : [];
    const count = Number.isFinite(c.count) ? Math.max(c.count, lines.length) : lines.length;
    return count > 0 ? { count, lines } : null;
  }

  /** "“a”, “b”, “c” and 2 more" (escaped HTML) for the flagged lines. */
  function confirmExamples(c) {
    const shown = c.lines.filter(Boolean).slice(0, CONFIRM_SHOWN);
    const more = c.count - shown.length;
    const quoted = shown.map((t) => `<q>${esc(t)}</q>`).join(", ");
    if (!quoted) return "";
    return more > 0 ? `${quoted} and ${more} more` : quoted;
  }

  /** Short sentence for the page turn / straightening the server did, or "". */
  function preText(pre) {
    if (!pre) return "";
    const out = [];
    const k = pre.quarterTurns;
    if (k === 1 || k === 2 || k === 3) {
      out.push(`The page was turned upright automatically (it was ${k === 2 ? "upside down" : "on its side"}).`);
    }
    if (typeof pre.deskewDegrees === "number" && Math.abs(pre.deskewDegrees) >= 0.1) {
      out.push(`Tilted text was straightened by ${Math.abs(pre.deskewDegrees).toFixed(1)} degrees.`);
    }
    return out.join(" ");
  }

  /** Words in a text (runs of non-space characters). */
  const wordCount = (text) => String(text || "").trim().split(/\s+/).filter(Boolean).length;
  const wordsText = (n) => `${n} ${n === 1 ? "word" : "words"}`;

  /**
   * Read the text on an image (Blob: JPEG, PNG or WebP, at most 10 MB). options.deskew asks the server
   * to straighten tilted text first. Resolves to { text, words (count), confidence (mean word confidence
   * 0..1, or null without words), seconds, models, pipeline, confirmNumbers ({ count, lines } or null),
   * preprocessing ({ quarterTurns, deskewDegrees } or null) }.
   * Rejects with an Error whose message can be shown to the user.
   */
  async function read(blob, signal, { deskew = false } = {}) {
    if (blob.size > MAX_BYTES) throw new Error(`this image is ${fmtBytes(blob.size)}; text can be read from images up to 10 MB.`);
    const form = new FormData();
    form.append("image", blob, "page");
    if (deskew) form.append("deskew", "1");
    let r;
    try {
      r = await fetch("api/ocr", { method: "POST", body: form, headers: { Accept: "application/json" }, signal });
    } catch (err) {
      if (err && err.name === "AbortError") throw err;
      throw new Error(`the local server did not answer. ${SERVER_HINT}`);
    }
    const data = await r.json().catch(() => null);
    if (!r.ok || !data || typeof data.text !== "string") {
      if (r.status === 404) throw new Error("reading text is not installed on this server.");
      if (r.status === 413) throw new Error("the image is too large (10 MB at most).");
      throw new Error((data && data.error) || `the server answered ${r.status}.`);
    }
    const conf = (data.words || []).map((w) => w.confidence).filter((c) => typeof c === "number");
    const flagged = flaggedLines(data.lines);
    const pre = data.preprocessing && typeof data.preprocessing === "object" ? data.preprocessing : null;
    return {
      text: data.text.slice(0, MAX_CHARS),
      words: conf.length || wordCount(data.text),
      confidence: conf.length ? conf.reduce((a, b) => a + b, 0) / conf.length : null,
      seconds: typeof data.seconds === "number" ? data.seconds : null,
      models: data.models || null,
      pipeline: data.pipeline === "v2" ? "v2" : "v1",
      confirmNumbers: flagged.length ? { count: flagged.length, lines: flagged.slice(0, CONFIRM_KEPT) } : null,
      preprocessing: pre ? { quarterTurns: Number(pre.quarter_turns) || 0,
        deskewDegrees: typeof pre.deskew_degrees === "number" ? pre.deskew_degrees : 0 } : null,
    };
  }

  /** Put text on the clipboard; falls back to selecting the textarea. Resolves to true when copied. */
  async function copy(text, textarea) {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch (err) { /* permission denied: try the old way */ }
    try {
      textarea.focus();
      textarea.select();
      return document.execCommand("copy");
    } catch (err) {
      return false;
    }
  }

  /**
   * Build a text box inside `root`. Options:
   *   id        prefix for element ids (unique on the page)
   *   heading   title of the box
   *   idleHint  one line shown before any text was read (a sentence on what the server can read is added)
   *   source    async () => Blob of the page image, or null when there is nothing to read; leave out when
   *             the document cannot be read (the Read text button is then not shown)
   *   prescription  show a reminder even when all numeric cues are missing from the transcription
   *   onRead    ({ text, ocr }) after a successful read
   *   onSave    async ({ text, ocr }) => {}; when given, a Save text button stores the user's corrections
   *   onBusy    (bool) reading started or ended
   * Returns { checkAvailability, setEnabled, setText, value, reset, pageChanged, isBusy }.
   */
  function widget(root, { id, heading = "Text", idleHint = "", source = null, prescription = false, onRead, onSave, onBusy }) {
    root.innerHTML = `<section class="ocr-box" aria-labelledby="${id}-h">
      <div class="ocr-head">
        <h4 class="h4" id="${id}-h">${esc(heading)}</h4>
        <div class="ocr-actions">
          <label class="ocr-option" hidden><input type="checkbox" data-ocr="deskew" id="${id}-deskew" />Straighten tilted text</label>
          <button type="button" class="btn" data-ocr="read" id="${id}-read" aria-describedby="${id}-hint"${source ? "" : " hidden"}>${icon("text")}<span>Read text</span></button>
        </div>
      </div>
      <p class="muted small ocr-hint" id="${id}-hint"></p>
      <p class="ocr-busy" hidden><span class="spinner small" aria-hidden="true"></span><span>Reading text... The first time can take 10 to 20 seconds.</span></p>
      <p class="visually-hidden" role="status" aria-live="polite" data-ocr="live"></p>
      <p class="notice warn ocr-error" role="alert" id="${id}-error" hidden></p>
      <div class="ocr-panel" hidden>
        <label for="${id}-text">Text read from the page (you can correct it)
          <textarea id="${id}-text" class="ocr-text" rows="8" maxlength="${MAX_CHARS}" spellcheck="false" autocapitalize="off" aria-describedby="${id}-confirm"></textarea></label>
        <p class="notice warn ocr-confirm" id="${id}-confirm" hidden></p>
        <p class="muted small ocr-pre" id="${id}-pre" hidden></p>
        <p class="notice ocr-note" id="${id}-note" hidden></p>
        <div class="ocr-foot">
          <span class="muted small ocr-count" id="${id}-count"></span>
          <span class="small ocr-done" role="status" data-ocr="done"></span>
          ${onSave ? `<button type="button" class="btn small primary" data-ocr="save" id="${id}-save" disabled>Save text</button>` : ""}
          <button type="button" class="btn small" data-ocr="copy" id="${id}-copy">${icon("copy")}Copy</button>
        </div>
      </div>
    </section>`;
    const el = {
      read: $('[data-ocr="read"]', root), hint: $(".ocr-hint", root), busy: $(".ocr-busy", root),
      live: $('[data-ocr="live"]', root), error: $(".ocr-error", root), panel: $(".ocr-panel", root),
      text: $(".ocr-text", root), note: $(".ocr-note", root), count: $(".ocr-count", root),
      done: $('[data-ocr="done"]', root), save: $('[data-ocr="save"]', root), copy: $('[data-ocr="copy"]', root),
      confirm: $(".ocr-confirm", root), pre: $(".ocr-pre", root), option: $(".ocr-option", root),
      deskew: $('[data-ocr="deskew"]', root),
    };
    const s = {
      available: null, reason: "", enabled: true, busy: false, seq: 0, ctl: null, caps: NO_CAPS,
      has: false, // the panel is shown (text was read or loaded)
      stale: false, // the page changed after the text was read
      info: null, // { confidence, words, readAt, models, confirmNumbers, ... } of the last read, or what was stored
      readText: "", // the text as it was read or last saved, to tell whether the user edited it
      doneTimer: null,
    };

    function update() {
      el.read.disabled = s.busy || !s.enabled || s.available !== true;
      $("span", el.read).textContent = s.has ? "Read again" : "Read text";
      let hint = "";
      if (source && s.available === false) hint = `Reading text is not available. ${s.reason}`;
      else if (s.stale) hint = "The page has changed since this text was read. Choose Read again to update it.";
      else if (!s.has && !s.busy) hint = idleHint && `${idleHint} ${capsText()}`.trim();
      el.hint.textContent = hint;
      el.option.hidden = !(source && s.available === true && s.caps.deskew);
      el.deskew.disabled = el.read.disabled;
      el.busy.hidden = !s.busy;
      el.panel.hidden = !s.has;
      const n = wordCount(el.text.value);
      el.count.textContent = wordsText(n);
      el.copy.disabled = !n;
      if (el.save) el.save.disabled = s.busy || el.text.value.trim() === s.readText;
      const c = s.info && typeof s.info.confidence === "number" ? s.info.confidence : null;
      let note = "", warn = false;
      if (s.has && s.info && s.info.words === 0 && !n) {
        note = "No text was found on this page. If it has printed text, try a sharper photo or the Colour or Grey output. You can also type the text here.";
      } else if (s.has && c !== null && c < LOW_CONFIDENCE && !s.info.edited && el.text.value.trim() === s.readText) {
        note = `Parts of this page were hard to read (average confidence ${Math.round(c * 100)}%). Check the text against the page and correct any mistakes.`;
        warn = true;
      }
      el.note.textContent = note;
      el.note.hidden = !note;
      el.note.classList.toggle("warn", warn);
      // Numeric warnings accompany confidence and orientation notes.
      const flagged = s.has ? confirmOf(s.info) : null;
      const edited = Boolean(s.info && s.info.edited) || el.text.value.trim() !== s.readText;
      let confirm = "";
      if (flagged && !edited) {
        const ex = confirmExamples(flagged);
        confirm = `<strong>Check the numbers.</strong> Numbers or doses were read${ex ? ` (for example: ${ex})` : ""}. `
          + "Always check numbers and units against the page.";
      } else if (flagged) {
        confirm = `<strong>Check the numbers.</strong> This text was corrected by hand. The original reading included numbers or doses `
          + `(${flagged.count} ${flagged.count === 1 ? "line" : "lines"}). Make sure every number and unit matches the page.`;
      }
      if (prescription && s.has) confirm += (confirm ? " " : "") + PRESCRIPTION_CHECK;
      el.confirm.innerHTML = confirm;
      el.confirm.hidden = !confirm;
      const pre = s.has && s.info ? preText(s.info.preprocessing) : "";
      el.pre.textContent = pre;
      el.pre.hidden = !pre;
    }

    /** One sentence on what this server reads, once its status is known. */
    function capsText() {
      if (s.available !== true) return "";
      return s.caps.handwriting
        ? "Handwriting is read too, and pages on their side or upside down are turned upright."
        : "Printed text only: handwriting is not read, and the page must be upright.";
    }

    function showError(msg) {
      el.error.textContent = msg;
      el.error.hidden = !msg;
    }

    function flash(msg) {
      clearTimeout(s.doneTimer);
      el.done.textContent = msg;
      if (msg) s.doneTimer = setTimeout(() => { el.done.textContent = ""; }, 4000);
    }

    function setBusy(on) {
      s.busy = on;
      update();
      if (onBusy) onBusy(on);
    }

    /** The text as it stands and what is known about it, or null when nothing is shown. */
    function value() {
      if (!s.has) return null;
      const text = el.text.value.trim().slice(0, MAX_CHARS);
      const info = s.info || {};
      return { text, ocr: {
        confidence: typeof info.confidence === "number" ? Math.round(info.confidence * 1000) / 1000 : null,
        words: wordCount(text),
        edited: Boolean(info.edited) || text !== s.readText,
        readAt: info.readAt || null,
        models: info.models || null,
        pipeline: info.pipeline || null,
        confirmNumbers: confirmOf(info),
        preprocessing: info.preprocessing || null,
      } };
    }

    /**
     * Show `text` in the panel. `info`: the stored { confidence, words, edited, readAt, models, pipeline,
     * confirmNumbers, preprocessing }, if any (older records have only the first five, or nothing).
     */
    function setText(text, info = null) {
      el.text.value = text || "";
      s.readText = el.text.value.trim();
      s.info = info;
      s.has = true;
      s.stale = false;
      update();
    }

    async function doRead() {
      if (s.busy || !source || el.read.disabled) return;
      const mine = ++s.seq;
      s.ctl = new AbortController();
      showError("");
      flash("");
      el.live.textContent = "Reading text...";
      setBusy(true);
      try {
        const blob = await source();
        if (!blob) throw new Error("there is no page to read.");
        const res = await read(blob, s.ctl.signal, { deskew: !el.option.hidden && el.deskew.checked });
        if (mine !== s.seq) return;
        setText(res.text, { confidence: res.confidence, words: res.words, edited: false,
          readAt: new Date().toISOString(), models: res.models, pipeline: res.pipeline,
          confirmNumbers: res.confirmNumbers, preprocessing: res.preprocessing });
        el.live.textContent = (res.words ? `Text read: ${wordsText(wordCount(res.text))}.` : "No text was found on this page.")
          + (res.confirmNumbers ? " Numbers or doses were read: check them against the page." : "")
          + (preText(res.preprocessing) ? ` ${preText(res.preprocessing)}` : "");
        if (onRead) await onRead(value());
      } catch (err) {
        if (mine !== s.seq || (err && err.name === "AbortError")) return;
        el.live.textContent = "";
        showError(`The text could not be read: ${err.message}`);
      } finally {
        if (mine === s.seq) setBusy(false);
      }
    }

    /** Forget the text and stop a read that is still running. */
    function reset() {
      s.seq++;
      if (s.ctl) s.ctl.abort();
      Object.assign(s, { has: false, stale: false, info: null, readText: "", ctl: null });
      el.text.value = "";
      el.live.textContent = "";
      showError("");
      flash("");
      if (s.busy) setBusy(false); else update();
    }

    /** The page under the text was replaced (new scan, other output mode, new corners). */
    function pageChanged() {
      if (s.busy) {
        s.seq++;
        if (s.ctl) s.ctl.abort();
        el.live.textContent = "";
        s.busy = false;
        if (onBusy) onBusy(false);
      }
      s.stale = s.has;
      showError("");
      update();
    }

    async function checkAvailability() {
      const r = await status();
      s.available = r.available;
      s.reason = r.reason;
      s.caps = r.capabilities || NO_CAPS;
      update();
      return r.available;
    }

    el.read.addEventListener("click", doRead);
    el.text.addEventListener("input", update);
    el.copy.addEventListener("click", async () => {
      const ok = await copy(el.text.value, el.text);
      flash(ok ? "Copied." : "Could not copy. Select the text and copy it yourself.");
    });
    if (el.save) {
      el.save.addEventListener("click", async () => {
        const v = value();
        el.save.disabled = true;
        showError("");
        try {
          await onSave(v);
          s.readText = v.text;
          s.info = v.ocr;
          flash("Saved.");
        } catch (err) {
          showError(`The text was not saved: ${err && err.message ? err.message : err}`);
        }
        update();
      });
    }
    update();

    return { checkAvailability, setEnabled: (on) => { s.enabled = on; update(); }, setText, value, reset, pageChanged,
      isBusy: () => s.busy, flash, showError };
  }

  return { MAX_BYTES, LOW_CONFIDENCE, PRESCRIPTION_CHECK, status, read, copy, wordCount, wordsText, confirmOf, widget };
})();

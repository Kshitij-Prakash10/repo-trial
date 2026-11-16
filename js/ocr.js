/*
 * Health Diary: reading the text on a page (OCR) with the local server.
 *
 *   GET  api/ocr/status  -> { available, reason?, handwriting }
 *   POST api/ocr (image) -> { text, lines: [{ text, confidence, script, low_confidence }],
 *                             words: [{ text, confidence, ... }], preprocessing: { quarter_turns } | null, ... }
 *
 * Ocr.widget(root, options) builds the "Text" box used in the Scan view and in the document viewer:
 * a Read text button, a progress line, and after reading an editable textarea with a word count, a Copy
 * button and a note when the text was hard to read. Right after a read it also says when the page was
 * turned before reading and, when some lines are handwritten or were hard to read, lists the lines as
 * read with a "Handwritten" or "Check" label on those. When the server cannot read text the button is
 * disabled and a one-line hint says why. Uses UI. English text: printed, and handwriting when the server
 * has the handwriting models; nothing leaves this computer.
 */
const Ocr = (() => {
  const { $, esc, icon, fmtBytes } = UI;
  const MAX_BYTES = 10 * 1024 * 1024; // bytes, the server's limit for one image
  const MAX_CHARS = 200000; // characters kept with a document
  const LOW_CONFIDENCE = 0.6; // mean word confidence (0..1) below which the text gets a "check it" note
  const SERVER_HINT = "Start the local server with `python server.py` and open http://127.0.0.1:8000.";

  /** Whether the server can read text: { available: bool, reason: string }. Never rejects. */
  async function status() {
    if (location.protocol === "file:") {
      return { available: false, reason: `It needs the local server: this page was opened as a file. ${SERVER_HINT}` };
    }
    let r;
    try {
      r = await fetch("api/ocr/status", { headers: { Accept: "application/json" }, cache: "no-store" });
    } catch (err) {
      return { available: false, reason: `The local server did not answer. ${SERVER_HINT}` };
    }
    const s = r.ok ? await r.json().catch(() => null) : null;
    if (s && s.available) return { available: true, reason: "", handwriting: s.handwriting === true };
    if (s) return { available: false, reason: s.reason || "The server cannot read text." };
    return { available: false, reason: r.status === 404
      ? "It is not installed on this server." : `The server answered ${r.status}.` };
  }

  /** Words in a text (runs of non-space characters). */
  const wordCount = (text) => String(text || "").trim().split(/\s+/).filter(Boolean).length;
  const wordsText = (n) => `${n} ${n === 1 ? "word" : "words"}`;

  /** Sentence for a page the server turned by `turns` quarter-turns (1..3) before reading, else "". */
  function turnedText(turns) {
    if (turns !== 1 && turns !== 2 && turns !== 3) return "";
    return `Page was rotated ${turns === 2 ? 180 : 90} degrees before reading.`;
  }

  /** Mean confidence (0..1) of the lines, each weighted by its number of words; null without lines. */
  function lineConfidence(lines) {
    let sum = 0, n = 0;
    for (const ln of lines) {
      if (typeof ln.confidence !== "number") continue;
      const w = wordCount(ln.text) || 1;
      sum += ln.confidence * w;
      n += w;
    }
    return n ? sum / n : null;
  }

  /**
   * Read the text on an image (Blob: JPEG, PNG or WebP, at most 10 MB). Resolves to
   * { text, words (count), confidence (mean confidence 0..1, or null without words), seconds, models,
   *   lines: [{ text, confidence, handwritten, low }], turns (quarter-turns the page was turned, 0..3),
   *   handwriting (the server read with the handwriting models) }.
   * Rejects with an Error whose message can be shown to the user.
   */
  async function read(blob, signal) {
    if (blob.size > MAX_BYTES) throw new Error(`this image is ${fmtBytes(blob.size)}; text can be read from images up to 10 MB.`);
    const form = new FormData();
    form.append("image", blob, "page");
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
    const lines = (Array.isArray(data.lines) ? data.lines : []).filter((ln) => ln && typeof ln.text === "string")
      .map((ln) => ({ text: ln.text, confidence: typeof ln.confidence === "number" ? ln.confidence : null,
        handwritten: ln.script === "handwritten", low: ln.low_confidence === true }));
    // A handwritten line keeps the handwriting reader's confidence, which the printed reader's words do
    // not have, so the average comes from the lines; an answer without lines falls back to the words.
    const byLine = lineConfidence(lines);
    const turns = data.preprocessing ? Number(data.preprocessing.quarter_turns) : 0;
    return {
      text: data.text.slice(0, MAX_CHARS),
      words: conf.length || wordCount(data.text),
      confidence: byLine !== null ? byLine : (conf.length ? conf.reduce((a, b) => a + b, 0) / conf.length : null),
      seconds: typeof data.seconds === "number" ? data.seconds : null,
      models: data.models || null,
      lines,
      turns: turns === 1 || turns === 2 || turns === 3 ? turns : 0,
      handwriting: data.handwriting === true,
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
   *   idleHint  one line shown before any text was read
   *   source    async () => Blob of the page image, or null when there is nothing to read; leave out when
   *             the document cannot be read (the Read text button is then not shown)
   *   onRead    ({ text, ocr }) after a successful read
   *   onSave    async ({ text, ocr }) => {}; when given, a Save text button stores the user's corrections
   *   onBusy    (bool) reading started or ended
   * Returns { checkAvailability, setEnabled, setText, value, reset, pageChanged, isBusy }.
   */
  function widget(root, { id, heading = "Text", idleHint = "", source = null, onRead, onSave, onBusy }) {
    root.innerHTML = `<section class="ocr-box" aria-labelledby="${id}-h">
      <div class="ocr-head">
        <h4 class="h4" id="${id}-h">${esc(heading)}</h4>
        <button type="button" class="btn" data-ocr="read" id="${id}-read" aria-describedby="${id}-hint"${source ? "" : " hidden"}>${icon("text")}<span>Read text</span></button>
      </div>
      <p class="muted small ocr-hint" id="${id}-hint"></p>
      <p class="ocr-busy" hidden><span class="spinner small" aria-hidden="true"></span><span>Reading text... The first time can take 10 to 20 seconds.</span></p>
      <p class="visually-hidden" role="status" aria-live="polite" data-ocr="live"></p>
      <p class="notice warn ocr-error" role="alert" id="${id}-error" hidden></p>
      <div class="ocr-panel" hidden>
        <label for="${id}-text">Text read from the page (you can correct it)
          <textarea id="${id}-text" class="ocr-text" rows="8" maxlength="${MAX_CHARS}" spellcheck="false" autocapitalize="off"></textarea></label>
        <p class="notice ocr-note" id="${id}-note" hidden></p>
        <p class="muted small ocr-turned" id="${id}-turned" hidden></p>
        <details class="ocr-lines" id="${id}-lines" hidden>
          <summary id="${id}-lines-sum"></summary>
          <ol class="ocr-line-list" tabindex="0" aria-labelledby="${id}-lines-sum"></ol>
        </details>
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
      turned: $(".ocr-turned", root), lines: $(".ocr-lines", root), lineSum: $(".ocr-lines summary", root),
      lineList: $(".ocr-line-list", root),
      done: $('[data-ocr="done"]', root), save: $('[data-ocr="save"]', root), copy: $('[data-ocr="copy"]', root),
    };
    const s = {
      available: null, reason: "", enabled: true, busy: false, seq: 0, ctl: null,
      has: false, // the panel is shown (text was read or loaded)
      stale: false, // the page changed after the text was read
      info: null, // { confidence, words, readAt, models } of the last read, or what was stored
      readText: "", // the text as it was read or last saved, to tell whether the user edited it
      detail: null, // { lines, turns } of the read just made; not stored with the document
      doneTimer: null,
    };

    function update() {
      el.read.disabled = s.busy || !s.enabled || s.available !== true;
      $("span", el.read).textContent = s.has ? "Read again" : "Read text";
      let hint = "";
      if (source && s.available === false) hint = `Reading text is not available. ${s.reason}`;
      else if (s.stale) hint = "The page has changed since this text was read. Choose Read again to update it.";
      else if (!s.has && !s.busy) hint = idleHint;
      el.hint.textContent = hint;
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
    }

    /** The parts that describe the read just made: the turned-page line and the lines as read. */
    function showDetail() {
      const d = s.has ? s.detail : null;
      const turned = d ? turnedText(d.turns) : "";
      el.turned.textContent = turned;
      el.turned.hidden = !turned;
      const lines = d ? d.lines : [];
      const hand = lines.filter((ln) => ln.handwritten).length;
      const low = lines.filter((ln) => ln.low).length;
      el.lines.hidden = !(hand || low);
      if (el.lines.hidden) {
        el.lineList.innerHTML = "";
        return;
      }
      const parts = [];
      if (hand) parts.push(`${hand} handwritten`);
      if (low) parts.push(`${low} to check`);
      el.lineSum.textContent = `Lines as read from the page (${lines.length}): ${parts.join(", ")}`;
      el.lineList.innerHTML = lines.map((ln) => {
        const pct = ln.confidence === null ? "" : ` (${Math.round(ln.confidence * 100)}%)`;
        const tags = (ln.handwritten ? `<span class="tag ocr-tag" title="Read as handwriting">${icon("edit")}Handwritten</span>` : "")
          + (ln.low ? `<span class="tag ocr-tag check" title="Low confidence${pct}: check this line against the page">${icon("eye")}Check${pct}</span>` : "");
        return `<li${ln.handwritten || ln.low ? ' class="flagged"' : ""}><span class="ocr-line-text">${esc(ln.text)}</span>${tags}</li>`;
      }).join("");
      el.lines.open = true;
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
      } };
    }

    /**
     * Show `text` in the panel. `info`: the stored { confidence, words, edited, readAt, models }, if any.
     * `detail`: { lines, turns } of a read just made (see read()); stored text has none.
     */
    function setText(text, info = null, detail = null) {
      el.text.value = text || "";
      s.readText = el.text.value.trim();
      s.info = info;
      s.detail = detail;
      s.has = true;
      s.stale = false;
      update();
      showDetail();
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
        const res = await read(blob, s.ctl.signal);
        if (mine !== s.seq) return;
        setText(res.text, { confidence: res.confidence, words: res.words, edited: false,
          readAt: new Date().toISOString(), models: res.models }, { lines: res.lines, turns: res.turns });
        const hand = res.lines.filter((ln) => ln.handwritten).length;
        const low = res.lines.filter((ln) => ln.low).length;
        el.live.textContent = (res.words ? `Text read: ${wordsText(wordCount(res.text))}.` : "No text was found on this page.")
          + (res.turns ? ` ${turnedText(res.turns)}` : "")
          + (hand ? ` ${hand} handwritten ${hand === 1 ? "line" : "lines"}.` : "")
          + (low ? ` ${low} ${low === 1 ? "line" : "lines"} to check.` : "");
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
      Object.assign(s, { has: false, stale: false, info: null, detail: null, readText: "", ctl: null });
      el.text.value = "";
      showDetail();
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

  return { MAX_BYTES, LOW_CONFIDENCE, status, read, copy, wordCount, wordsText, turnedText, widget };
})();

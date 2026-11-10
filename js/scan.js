/*
 * Health Diary: the Scan view (#scan). Three steps: 1 Photo, 2 Check, 3 Save.
 * Sends the photo to the local server (POST /api/scan), shows the detected outline over the photo and the
 * corrected page, lets the user switch the output mode, drag the corners and re-warp (field `corners`),
 * optionally reads the text on the page (Ocr.widget, POST /api/ocr on the corrected page), then saves the
 * page as a PNG document together with that text. Without the scanner the photo can still be saved as it is.
 * Uses UI, FilesDB, Ocr, Documents. Interface: docs/PLAN.md ("Interface: the scanner").
 */
const Scan = (() => {
  const { $, $$, fmtBytes, fmtDate, today } = UI;
  const MAX_PHOTO = 10 * 1024 * 1024; // bytes, the scanner's limit
  const SERVER_HINT = "Start the local server with `python server.py` and open http://127.0.0.1:8000.";
  const STEPS = ["photo", "check", "save"];
  const HEADINGS = { photo: "#scan-h-photo", check: "#scan-h-check", save: "#scan-h-save", done: "#scan-h-done" };
  const CORNER_NAMES = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"];
  const MODE_TEXT = { color: "Colour", gray: "Grey", bw: "Black and white" };
  const KEY_STEP = 0.005, KEY_STEP_BIG = 0.05; // fraction of the photo's width or height per arrow-key press

  const st = {
    canScan: false, // the server can scan (even without a model: whole-photo clean-up)
    statusCheck: null, // pending checkStatus() promise
    step: "photo",
    file: null, photoUrl: null, photoW: 0, photoH: 0, // photo px after EXIF orientation
    result: null, // last /api/scan response + { mode }
    manual: null, // corners (px) last applied by the user; re-sent on a mode switch
    raw: false, // saving the photo itself, not a scan
    adjusting: false, edit: null, // while adjusting: normalised corner handles [[x, y] * 4]
    busy: false, seq: 0, fieldsReady: false,
  };

  let ocr = null; // the "Text on this page" box of the Check step (Ocr.widget), made in bind()

  const mode = () => ($('input[name="scan-mode"]:checked') || {}).value || "gray";
  const setStatus = (msg) => { $("#scan-status").textContent = msg; };
  const setPhotoStatus = (msg, offerRaw = false) => {
    $("#scan-photo-status").textContent = msg;
    $("#scan-raw-row").hidden = !offerRaw;
  };
  const setAvail = (msg, kind = "") => {
    const el = $("#scan-avail");
    el.textContent = msg;
    el.className = `notice ${kind}`;
  };
  const pageSrc = (r) => "data:image/png;base64," + r.image_png_b64;

  /* ---------- Steps ---------- */

  /** Show one step ("photo" | "check" | "save" | "done"), update the step indicator, focus its heading. */
  function setStep(step, focus = true) {
    st.step = step;
    $("#scan-step-photo").hidden = step !== "photo";
    $("#scan-step-check").hidden = step !== "check";
    $("#scan-save").hidden = step !== "save";
    $("#scan-done").hidden = step !== "done";
    const at = step === "done" ? STEPS.length : STEPS.indexOf(step);
    $$("#scan-steps li").forEach((li, i) => {
      const skipped = li.dataset.step === "check" && st.raw && at > i;
      const state = i < at ? (skipped ? "skipped" : "done") : i === at ? "current" : "";
      li.className = state;
      if (state === "current") li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
      $(".step-state", li).textContent = state === "done" ? " (done)" : state === "skipped" ? " (skipped)" : "";
    });
    if (focus) $(HEADINGS[step]).focus();
  }

  /* ---------- Scanner availability ---------- */

  const NO_SCAN_TAIL = "You can still save a photo of the page as a document without scanning.";

  async function queryStatus() {
    const model = $("#scan-model");
    if (location.protocol === "file:") {
      setAvail(`Scanning needs the local server: this page was opened as a file. ${SERVER_HINT} ${NO_SCAN_TAIL}`, "warn");
      model.textContent = "";
      return (st.canScan = false);
    }
    if (!st.canScan) setAvail("Checking the scanner...");
    let s = null, code = 0;
    try {
      const r = await fetch("api/scan/status", { headers: { Accept: "application/json" }, cache: "no-store" });
      code = r.status;
      if (r.ok) s = await r.json().catch(() => null);
    } catch (err) {
      code = -1;
    }
    if (code === -1) {
      setAvail(`The scanner server did not answer. ${SERVER_HINT} ${NO_SCAN_TAIL}`, "warn");
      st.canScan = false;
    } else if (!s) {
      setAvail(code === 404
        ? `The scanner is not installed on this server, so scanning is not available. ${NO_SCAN_TAIL}`
        : `The scanner is not available (the server answered ${code}). ${NO_SCAN_TAIL}`, "warn");
      st.canScan = false;
    } else if (s.available) {
      setAvail("");
      st.canScan = true;
    } else {
      setAvail("The page-detection model is not installed on this server. A scan will only clean up the whole photo; it will not find the page edges, but you can set the corners yourself.", "warn");
      st.canScan = true;
    }
    model.textContent = !s ? "" : s.available && s.model_version
      ? `Page detection model: ${s.model_version}. Photos are processed on this computer and not kept by the server.`
      : "No page detection model is installed.";
    return st.canScan;
  }

  /** Ask the server whether the scanner is there. Resolves to true when scans can be sent. */
  function checkStatus() {
    const p = queryStatus().finally(() => { if (st.statusCheck === p) st.statusCheck = null; });
    st.statusCheck = p;
    return p;
  }

  /* ---------- Photo ---------- */

  function clearPhoto() {
    st.seq++;
    endAdjust(false);
    if (st.photoUrl) URL.revokeObjectURL(st.photoUrl);
    Object.assign(st, { file: null, photoUrl: null, photoW: 0, photoH: 0, result: null, manual: null, raw: false,
      fieldsReady: false });
    $("#scan-photo").removeAttribute("src");
    $("#scan-page").removeAttribute("src");
    $("#scan-save-preview").removeAttribute("src");
    drawOutline(null);
    if (ocr) ocr.reset();
    $("#scan-save-text").hidden = true;
    $("#scan-notfound").hidden = true;
    $(".scan-page-wrap").classList.add("empty");
    setBusy(false);
    setStatus("");
    $("#scan-status").classList.remove("visually-hidden");
    $("#scan-meta").textContent = "";
    $("#scan-save-status").textContent = "";
    $("#scan-camera").value = "";
    $("#scan-file").value = "";
    $('input[name="scan-mode"][value="gray"]').checked = true;
  }

  /** Back to step 1 with nothing loaded. */
  function reset(focus = false) {
    clearPhoto();
    setPhotoStatus("");
    setStep("photo", focus);
  }

  async function onPick(f) {
    if (!f) return;
    clearPhoto();
    setPhotoStatus("");
    const type = await FilesDB.sniffDocument(f);
    if (!type || !type.startsWith("image/")) {
      setPhotoStatus(type === "application/pdf"
        ? "This is a PDF. The scanner needs a photo (JPEG, PNG or WebP). To store a PDF, use Add a document in Documents."
        : "The scanner needs a JPEG, PNG or WebP photo. (HEIC photos from some phones are not supported: set the camera to \"Most compatible\" or choose a JPEG.)");
      return;
    }
    if (f.size > FilesDB.MAX_BYTES) {
      setPhotoStatus(`This photo is ${fmtBytes(f.size)}. Documents can be up to 20 MB.`);
      return;
    }
    st.file = f;
    st.photoUrl = URL.createObjectURL(f);
    const img = $("#scan-photo");
    img.src = st.photoUrl;
    try { await img.decode(); } catch (err) { /* the size is taken from the scan result instead */ }
    if (st.file !== f) return; // another photo was picked meanwhile
    st.photoW = img.naturalWidth; st.photoH = img.naturalHeight;
    if (st.statusCheck) await st.statusCheck;
    if (st.file !== f) return;
    if (!st.canScan) { goRaw(); return; }
    if (f.size > MAX_PHOTO) {
      setPhotoStatus(`This photo is ${fmtBytes(f.size)}; the scanner takes photos up to 10 MB. Choose a smaller photo, or save this one without scanning.`, true);
      return;
    }
    setStep("check");
    run();
  }

  /* ---------- Scan ---------- */

  function setBusy(on) {
    st.busy = on;
    $("#scan-step-check").classList.toggle("is-busy", on);
    if (on) $("#scan-result").setAttribute("aria-busy", "true"); else $("#scan-result").removeAttribute("aria-busy");
    $$('input[name="scan-mode"]').forEach((r) => { r.disabled = on; });
    ["#scan-adjust", "#scan-apply", "#scan-nf-adjust"].forEach((s) => { $(s).disabled = on; });
    if (ocr) ocr.setEnabled(!on);
    lockNext();
  }

  /** Continue waits for a scan or a text read that is still running, so the text is not lost. */
  function lockNext() {
    const wait = st.busy || Boolean(ocr && ocr.isBusy());
    ["#scan-next", "#scan-nf-save"].forEach((s) => { $(s).disabled = wait; });
  }

  /** POST the photo (and the user's corners, if any) to /api/scan and show the result. */
  async function run() {
    if (!st.file || st.busy) return;
    const mine = ++st.seq;
    const m = mode();
    setBusy(true);
    setStatus(st.manual ? "Straightening the page with your corners..." : "Finding the page...");
    const form = new FormData();
    form.append("image", st.file, st.file.name || "photo.jpg");
    form.append("mode", m);
    if (st.manual) form.append("corners", JSON.stringify(st.manual));
    try {
      let r;
      try {
        r = await fetch("api/scan", { method: "POST", body: form, headers: { Accept: "application/json" } });
      } catch (err) {
        throw new Error(`the scanner server did not answer. ${SERVER_HINT}`);
      }
      const data = await r.json().catch(() => null);
      if (!r.ok || !data) {
        if (r.status === 413) throw new Error("the photo is too large for the scanner (10 MB at most).");
        if (r.status === 404) throw new Error("the scanner is not installed on this server.");
        throw new Error((data && data.error) || `the server answered ${r.status}.`);
      }
      if (mine !== st.seq) return; // retake or reset meanwhile
      const wasAdjusting = st.adjusting;
      const first = !st.result;
      st.result = { ...data, mode: m };
      if (ocr && !first) ocr.pageChanged(); // text read from the earlier page is kept but marked as out of date
      setBusy(false);
      endAdjust(false);
      show(st.result);
      if (wasAdjusting) $("#scan-adjust").focus();
    } catch (err) {
      if (mine !== st.seq) return;
      setBusy(false);
      if (!st.result) { // nothing to check: back to the photo step, offer to save the photo as it is
        setStep("photo");
        setPhotoStatus(`The scan did not work: ${err.message} You can try another photo, or save this one without scanning.`, true);
        return;
      }
      $(`input[name="scan-mode"][value="${st.result.mode}"]`).checked = true; // the page shown is still in that mode
      if (st.adjusting) st.manual = st.result.method === "manual" ? st.result.corners : null;
      setStatus(`The scan did not work: ${err.message}`);
    }
  }

  function confidenceText(c) {
    const p = `${Math.round(c * 100)}%`;
    if (c >= 0.8) return `The page was found with high confidence (${p}). Check the corrected page, then continue.`;
    if (c >= 0.6) return `The page was found with medium confidence (${p}). Check that the outline follows the page edges; if not, adjust the corners.`;
    return `The page was found, but with low confidence (${p}). Check the outline carefully and adjust the corners if needed.`;
  }

  function show(r) {
    $("#scan-page").src = pageSrc(r);
    $(".scan-page-wrap").classList.remove("empty");
    if (!st.photoW && !r.page_found && Array.isArray(r.corners) && r.corners.length === 4) {
      st.photoW = r.corners[2][0]; st.photoH = r.corners[2][1]; // whole-photo corners give W x H
    }
    drawOutline(r.page_found && Array.isArray(r.corners_norm) && r.corners_norm.length === 4 ? r.corners_norm : null);
    $("#scan-notfound").hidden = Boolean(r.page_found);
    // The not-found notice says it all on screen; the status line still announces it to screen readers.
    $("#scan-status").classList.toggle("visually-hidden", !r.page_found);
    let msg;
    if (r.method === "manual") msg = "The page was straightened with your corners. Check it, then continue.";
    else if (!r.page_found) msg = "The page edges could not be found, so the whole photo is shown.";
    else if (typeof r.confidence === "number") msg = confidenceText(r.confidence);
    else msg = "The page was found. Check the corrected page, then continue.";
    setStatus(msg);
    $("#scan-photo-cap").textContent = r.page_found
      ? (r.method === "manual" ? "Your photo with your corners outlined." : "Your photo with the detected page outlined.")
      : "Your photo. No page outline was found.";
    const how = r.method === "manual" ? "your corners" : r.method === "ml" ? "trained model" : "whole photo";
    $("#scan-meta").textContent = `Corrected page: ${MODE_TEXT[r.mode] || r.mode}, ${r.width} x ${r.height} px (${how}${r.elapsed_ms != null ? `, ${Math.round(r.elapsed_ms)} ms` : ""}).`;
  }

  /* ---------- Outline and corner handles ---------- */

  function drawOutline(pts) {
    $("#scan-outline polygon").setAttribute("points", pts ? pts.map(([x, y]) => `${x},${y}`).join(" ") : "");
  }

  const pct = (v) => `${Math.round(v * 100)}%`;
  function placeHandle(btn, i) {
    const [x, y] = st.edit[i];
    btn.style.left = `${(x * 100).toFixed(3)}%`;
    btn.style.top = `${(y * 100).toFixed(3)}%`;
    btn.setAttribute("aria-label", `${CORNER_NAMES[i]} corner, ${pct(x)} across, ${pct(y)} down`);
  }

  function startAdjust() {
    if (!st.result || st.busy || st.adjusting) return;
    const r = st.result;
    const clamp = (v) => Math.min(1, Math.max(0, v));
    st.edit = r.page_found && Array.isArray(r.corners_norm) && r.corners_norm.length === 4
      ? r.corners_norm.map(([x, y]) => [clamp(x), clamp(y)])
      : [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]];
    st.adjusting = true;
    const wrap = $("#scan-photo-wrap");
    wrap.classList.add("adjusting");
    st.edit.forEach((_, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "corner-handle";
      b.dataset.i = String(i);
      b.setAttribute("aria-describedby", "scan-adjust-help");
      placeHandle(b, i);
      wrap.appendChild(b);
    });
    drawOutline(st.edit);
    $("#scan-adjust-tools").hidden = false;
    $("#scan-adjust").hidden = true;
    $("#scan-adjust").setAttribute("aria-expanded", "true");
    $("#scan-photo-cap").textContent = "Drag the handles onto the corners of the page, then apply.";
    $(".corner-handle", wrap).focus();
  }

  /** Leave corner editing. `restore`: put the outline and captions back to the result shown. */
  function endAdjust(restore = true) {
    $$(".corner-handle").forEach((h) => h.remove());
    $("#scan-photo-wrap").classList.remove("adjusting");
    $("#scan-adjust-tools").hidden = true;
    $("#scan-adjust").hidden = false;
    $("#scan-adjust").setAttribute("aria-expanded", "false");
    const was = st.adjusting;
    st.adjusting = false; st.edit = null;
    if (was && restore && st.result) show(st.result);
  }

  function moveHandle(i, x, y) {
    st.edit[i] = [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
    const b = $(`.corner-handle[data-i="${i}"]`);
    if (b) placeHandle(b, i);
    drawOutline(st.edit);
  }

  /** True when the quad (TL, TR, BR, BL) is strictly convex, so it can be warped. */
  function isConvex(q) {
    const s = q.map((p, i) => {
      const a = q[(i + 1) % 4], b = q[(i + 2) % 4];
      return (a[0] - p[0]) * (b[1] - a[1]) - (a[1] - p[1]) * (b[0] - a[0]);
    });
    return s.every((v) => v > 0) || s.every((v) => v < 0);
  }

  function apply() {
    if (!st.adjusting || st.busy) return;
    const w = st.photoW, h = st.photoH;
    if (!w || !h) { setStatus("The photo size is not known yet. Try again in a moment."); return; }
    const px = st.edit.map(([x, y]) => [Math.round(x * w * 10) / 10, Math.round(y * h * 10) / 10]);
    if (!isConvex(px)) {
      setStatus("The corners cross over or fold in. Move them so that they outline the page, then apply again.");
      return;
    }
    st.manual = px;
    run();
  }

  /* ---------- Save ---------- */

  function fillFields(title) {
    if (st.fieldsReady) return;
    $("#scan-save [data-fields]").innerHTML = Documents.fieldsHtml("scan", { title, docDate: today() });
    st.fieldsReady = true;
  }

  /** Step 3 with the corrected page. */
  function goSave() {
    if (!st.result || st.busy) return;
    if (st.adjusting) endAdjust();
    st.raw = false;
    fillFields(`Scanned document ${fmtDate(today())}`);
    $("#scan-save-preview").src = pageSrc(st.result);
    $("#scan-save-cap").textContent = `Corrected page, ${MODE_TEXT[st.result.mode] || st.result.mode}, saved as PNG.`;
    $("#scan-save-status").textContent = "";
    const t = ocr ? ocr.value() : null;
    const note = $("#scan-save-text");
    note.hidden = !(t && t.text);
    note.textContent = t && t.text
      ? `The text you read from the page (${Ocr.wordsText(t.ocr.words)}) will be saved with the document. Go Back to change it.` : "";
    setStep("save");
  }

  /** Step 3 with the photo as it is (no scanner, or the user chose not to scan). */
  function goRaw() {
    if (!st.file) return;
    st.seq++; // ignore a scan still running
    setBusy(false);
    st.raw = true;
    fillFields(`Document photo ${fmtDate(today())}`);
    $("#scan-save-preview").src = st.photoUrl;
    $("#scan-save-cap").textContent = "Your photo, saved as it is (not scanned).";
    $("#scan-save-text").hidden = true;
    $("#scan-save-status").textContent = "";
    setPhotoStatus("");
    setStep("save");
  }

  async function save(e) {
    e.preventDefault();
    const btn = $("#scan-save-btn");
    if (btn.disabled || (st.raw ? !st.file : !st.result)) return;
    btn.disabled = true;
    const status = $("#scan-save-status");
    status.textContent = "Saving...";
    try {
      const meta = Documents.readFields(e.target);
      let blob;
      if (st.raw) {
        blob = st.file;
        meta.title = meta.title || "Document photo";
        meta.fileName = st.file.name || "photo";
      } else {
        const r = st.result;
        blob = UI.b64ToBlob(r.image_png_b64, "image/png");
        meta.title = meta.title || "Scanned document";
        meta.fileName = `${meta.title.replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "-") || "scan"}.png`;
        meta.source = "scan";
        meta.scan = { mode: r.mode, method: r.method, modelVersion: r.model_version || null,
          confidence: r.confidence ?? null, pageFound: Boolean(r.page_found) };
        const t = ocr ? ocr.value() : null;
        if (t && t.text) { meta.text = t.text; meta.ocr = t.ocr; }
      }
      meta.thumb = await UI.thumbnail(blob);
      const d = await Documents.add(meta, blob);
      $("#scan-done-text").textContent = `"${d.title}" was saved (${FilesDB.DOC_TYPES[d.mime] || d.mime}, ${fmtBytes(d.size)}).`;
      const wasRaw = st.raw;
      clearPhoto();
      st.raw = wasRaw; // the step indicator shows Check as skipped
      setStep("done");
      st.raw = false;
    } catch (err) {
      status.textContent = err.code ? `Not saved: ${err.message}` : `Not saved. ${Documents.storageError(err)}`;
    } finally {
      btn.disabled = false;
    }
  }

  /* ---------- Events ---------- */

  function bindHandles() {
    const wrap = $("#scan-photo-wrap");
    let drag = null; // { i, id }
    wrap.addEventListener("pointerdown", (e) => {
      const h = e.target.closest(".corner-handle");
      if (!h || !st.adjusting) return;
      e.preventDefault();
      h.focus();
      h.setPointerCapture(e.pointerId);
      drag = { i: Number(h.dataset.i), id: e.pointerId };
      h.classList.add("dragging");
    });
    wrap.addEventListener("pointermove", (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      const r = $("#scan-photo").getBoundingClientRect();
      moveHandle(drag.i, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
    });
    const end = (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      const h = $(`.corner-handle[data-i="${drag.i}"]`);
      if (h) h.classList.remove("dragging");
      drag = null;
    };
    wrap.addEventListener("pointerup", end);
    wrap.addEventListener("pointercancel", end);
    wrap.addEventListener("keydown", (e) => {
      const h = e.target.closest(".corner-handle");
      if (!h || !st.adjusting) return;
      if (e.key === "Escape") { e.preventDefault(); endAdjust(); $("#scan-adjust").focus(); return; }
      if (e.key === "Enter") { e.preventDefault(); if (!e.repeat) apply(); return; }
      const d = e.shiftKey ? KEY_STEP_BIG : KEY_STEP;
      const mv = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, -d], ArrowDown: [0, d] }[e.key];
      if (!mv) return;
      e.preventDefault();
      const i = Number(h.dataset.i);
      moveHandle(i, st.edit[i][0] + mv[0], st.edit[i][1] + mv[1]);
    });
  }

  function bindDrop() {
    const zone = $("#scan-drop"), panel = $("#scan-step-photo");
    const hasFiles = (e) => e.dataTransfer && [...(e.dataTransfer.types || [])].includes("Files");
    panel.addEventListener("dragover", (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
      zone.classList.add("over");
    });
    panel.addEventListener("dragleave", (e) => {
      if (!panel.contains(e.relatedTarget)) zone.classList.remove("over");
    });
    panel.addEventListener("drop", (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      zone.classList.remove("over");
      onPick(e.dataTransfer.files[0]);
    });
  }

  function bind() {
    $("#scan-camera").addEventListener("change", (e) => onPick(e.target.files[0]));
    $("#scan-file").addEventListener("change", (e) => onPick(e.target.files[0]));
    $("#scan-raw-btn").addEventListener("click", goRaw);
    $$('input[name="scan-mode"]').forEach((r) => r.addEventListener("change", () => { if (st.file) run(); }));
    const retake = () => reset(true);
    $("#scan-retake").addEventListener("click", retake);
    $("#scan-nf-retake").addEventListener("click", retake);
    $("#scan-adjust").addEventListener("click", startAdjust);
    $("#scan-nf-adjust").addEventListener("click", startAdjust);
    $("#scan-adjust-cancel").addEventListener("click", () => { endAdjust(); $("#scan-adjust").focus(); });
    $("#scan-apply").addEventListener("click", apply);
    $("#scan-next").addEventListener("click", goSave);
    $("#scan-nf-save").addEventListener("click", goSave);
    $("#scan-back").addEventListener("click", () => {
      if (st.raw && !st.result) { reset(true); return; }
      st.raw = false;
      setStep("check");
    });
    $("#scan-discard").addEventListener("click", () => { reset(true); setPhotoStatus("Discarded. Take or choose another photo."); });
    $("#scan-save").addEventListener("submit", save);
    $("#scan-another").addEventListener("click", () => reset(true));
    $("#scan-view-docs").addEventListener("click", () => reset(false));
    bindHandles();
    bindDrop();
    if (typeof Ocr !== "undefined") {
      ocr = Ocr.widget($("#scan-ocr"), {
        id: "scan-ocr", heading: "Text on this page",
        idleHint: "Read the printed text on the page so you can copy it and find this document later by its words. It is read on this computer and saved with the document.",
        source: async () => (st.result ? UI.b64ToBlob(st.result.image_png_b64, "image/png") : null),
        onBusy: lockNext,
      });
    }
    document.addEventListener("view:show", (e) => {
      if (e.detail.name !== "scan") return;
      if (st.step === "done") reset(false);
      if (ocr) ocr.checkAvailability();
      if (st.step === "photo") {
        if (!st.file) setPhotoStatus("");
        checkStatus();
      }
    });
    document.addEventListener("user:open", () => reset(false));
    document.addEventListener("user:close", () => reset(false));
  }

  bind();
  return { checkStatus, reset };
})();

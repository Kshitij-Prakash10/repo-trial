/*
 * Health Diary: scan a paper document. Sends the photo to the local server (POST /api/scan), shows the
 * detected page outline over the photo and the corrected page, and saves the page as a PNG document.
 * Uses UI, FilesDB, Documents. Interface: docs/PLAN.md ("Interface: the scanner").
 */
const Scan = (() => {
  const { $, $$, esc, fmtBytes, toast, today } = UI;
  const MAX_PHOTO = 10 * 1024 * 1024; // bytes, the scanner's limit
  const SERVER_HINT = "Start the local server with `python server.py` and open http://127.0.0.1:8000.";

  let file = null, photoUrl = null, result = null, busy = false, available = false, seq = 0;

  const mode = () => ($('input[name="scan-mode"]:checked') || {}).value || "gray";
  const setAvail = (msg, kind = "") => {
    const el = $("#scan-avail");
    el.textContent = msg;
    el.className = `notice ${kind}`;
  };
  const setStatus = (msg) => { $("#scan-status").textContent = msg; };

  /** Ask the server whether the scanner is there. Shows the result; resolves to true when scans can be sent. */
  async function checkStatus() {
    if (location.protocol === "file:") {
      setAvail(`Scanning needs the local server: this page was opened as a file. ${SERVER_HINT} Until then you can add a photo of the page with Add a document.`, "warn");
      available = false;
      $("#scan-body").hidden = true;
      return false;
    }
    setAvail("Checking the scanner...");
    let s = null, code = 0;
    try {
      const r = await fetch("api/scan/status", { headers: { Accept: "application/json" }, cache: "no-store" });
      code = r.status;
      if (r.ok) s = await r.json().catch(() => null);
    } catch (err) {
      code = -1;
    }
    if (code === -1) {
      setAvail(`The scanner server did not answer. ${SERVER_HINT}`, "warn");
      available = false;
    } else if (!s) {
      setAvail(code === 404
        ? "The scanner is not installed on this server, so scanning is not available. You can still add a photo of the page with Add a document."
        : `The scanner is not available (the server answered ${code}). You can still add a photo of the page with Add a document.`, "warn");
      available = false;
    } else if (s.available) {
      setAvail(`Scanner ready${s.model_version ? ` (model ${s.model_version})` : ""}. The photo is processed on this computer and not stored by the server.`, "ok");
      available = true;
    } else {
      setAvail("The page-detection model is not installed on this server. A scan will only enhance the whole photo; it will not find the page edges.", "warn");
      available = true;
    }
    $("#scan-body").hidden = !available;
    return available;
  }

  function reset() {
    seq++;
    file = null; result = null;
    if (photoUrl) URL.revokeObjectURL(photoUrl);
    photoUrl = null;
    $("#scan-result").hidden = true;
    $("#scan-save").hidden = true;
    $("#scan-photo").removeAttribute("src");
    $("#scan-page").removeAttribute("src");
    $("#scan-outline polygon").setAttribute("points", "");
    $$(".corner", $(".scan-photo-wrap")).forEach((c) => c.remove());
    setStatus("");
    $("#scan-camera").value = "";
    $("#scan-file").value = "";
  }

  function open() {
    const panel = $("#doc-scan");
    panel.hidden = false;
    $("#doc-scan-btn").setAttribute("aria-expanded", "true");
    $("#doc-upload").hidden = true;
    $("#doc-add-btn").setAttribute("aria-expanded", "false");
    $("#scan-h").focus();
    checkStatus();
  }

  function close() {
    reset();
    $("#doc-scan").hidden = true;
    $("#doc-scan-btn").setAttribute("aria-expanded", "false");
  }

  async function onPick(input) {
    const f = input.files[0];
    if (!f) return;
    reset();
    if (f.size > MAX_PHOTO) {
      setStatus(`This photo is ${fmtBytes(f.size)}. The scanner takes photos up to 10 MB.`);
      return;
    }
    const type = await FilesDB.sniffDocument(f);
    if (!type || !type.startsWith("image/")) {
      setStatus("The scanner needs a JPEG, PNG or WebP photo. (HEIC photos from some phones are not supported: set the camera to \"Most compatible\" or choose a JPEG.)");
      return;
    }
    file = f;
    photoUrl = URL.createObjectURL(f);
    $("#scan-photo").src = photoUrl;
    run();
  }

  async function run() {
    if (!file || busy) return;
    busy = true;
    const mine = ++seq;
    $("#scan-mode-set").setAttribute("aria-busy", "true");
    $$('input[name="scan-mode"]').forEach((r) => { r.disabled = true; });
    setStatus("Finding the page...");
    const form = new FormData();
    form.append("image", file, file.name || "photo.jpg");
    form.append("mode", mode());
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
      if (mine !== seq) return; // reset or another scan started meanwhile
      result = { ...data, mode: mode() };
      show(result);
    } catch (err) {
      if (mine === seq) setStatus(`The scan did not work: ${err.message}`);
    } finally {
      busy = false;
      $("#scan-mode-set").removeAttribute("aria-busy");
      $$('input[name="scan-mode"]').forEach((r) => { r.disabled = false; });
      // A new photo was picked while this scan was running: scan that one now.
      if (mine !== seq && file && !result) run();
    }
  }

  function show(r) {
    $("#scan-page").src = "data:image/png;base64," + r.image_png_b64;
    const wrap = $(".scan-photo-wrap");
    $$(".corner", wrap).forEach((c) => c.remove());
    const pts = Array.isArray(r.corners_norm) ? r.corners_norm : [];
    const poly = $("#scan-outline polygon");
    if (r.page_found && pts.length === 4) {
      poly.setAttribute("points", pts.map(([x, y]) => `${x},${y}`).join(" "));
      pts.forEach(([x, y]) => {
        const c = document.createElement("span");
        c.className = "corner";
        c.style.left = `${(x * 100).toFixed(3)}%`;
        c.style.top = `${(y * 100).toFixed(3)}%`;
        wrap.appendChild(c);
      });
      $("#scan-photo-cap").textContent = "Your photo with the detected page outlined.";
    } else {
      poly.setAttribute("points", "");
      $("#scan-photo-cap").textContent = "Your photo. No page outline was found.";
    }
    const conf = typeof r.confidence === "number" ? ` Confidence ${(r.confidence * 100).toFixed(0)}%.` : "";
    setStatus(r.page_found
      ? `Page found (${r.method === "ml" ? "trained model" : "fallback method"}).${conf} Check the corrected page, then save it.`
      : "No page edge was found, so the whole photo was enhanced. Try a darker background, or move closer so the page fills the photo.");
    $("#scan-meta").textContent = `Corrected page, ${r.width} x ${r.height} px${r.elapsed_ms != null ? `, ${Math.round(r.elapsed_ms)} ms` : ""}.`;
    $("#scan-result").hidden = false;
    const form = $("#scan-save");
    if (form.hidden) {
      form.querySelector("[data-fields]").innerHTML = Documents.fieldsHtml("scan", { title: "", docDate: today() });
      form.hidden = false;
    }
  }

  async function save(e) {
    e.preventDefault();
    if (!result) return;
    const btn = $("#scan-save-btn");
    btn.disabled = true;
    try {
      const blob = UI.b64ToBlob(result.image_png_b64, "image/png");
      const meta = Documents.readFields(e.target);
      meta.title = meta.title || "Scanned document";
      meta.fileName = `${meta.title.replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "-") || "scan"}.png`;
      meta.thumb = await UI.thumbnail(blob);
      meta.source = "scan";
      meta.scan = { mode: result.mode, method: result.method, modelVersion: result.model_version || null,
        confidence: result.confidence ?? null, pageFound: Boolean(result.page_found) };
      const d = await Documents.add(meta, blob);
      toast(`Saved "${d.title}" (${fmtBytes(d.size)}).`);
      close();
      $("#doc-scan-btn").focus();
    } catch (err) {
      setStatus(err.code ? `Not saved: ${err.message}` : `Not saved. ${Documents.storageError(err)}`);
    } finally {
      btn.disabled = false;
    }
  }

  function bind() {
    $("#doc-scan-btn").addEventListener("click", () => ($("#doc-scan").hidden ? open() : close()));
    $("#scan-close").addEventListener("click", () => { close(); $("#doc-scan-btn").focus(); });
    $("#scan-discard").addEventListener("click", () => { reset(); setStatus("Scan discarded. Take or choose another photo."); });
    $("#scan-camera").addEventListener("change", (e) => onPick(e.target));
    $("#scan-file").addEventListener("change", (e) => onPick(e.target));
    $$('input[name="scan-mode"]').forEach((r) => r.addEventListener("change", () => { if (file) run(); }));
    $("#scan-save").addEventListener("submit", save);
    document.addEventListener("user:open", close);
  }

  bind();
  return { open, close, checkStatus };
})();

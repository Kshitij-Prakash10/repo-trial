/* Health Diary: voice notes (record with MediaRecorder or upload a file; list, play, rename, delete). */
const Voice = (() => {
  const { $, $$, esc, icon, fmtDate, fmtBytes, fmtDuration, toast, emit } = UI;
  const MAX_SECONDS = 15 * 60; // a recording stops by itself after 15 minutes

  let userId = null, notes = [], loaded = false, loadError = null, preselect = "";
  let recorder = null, stream = null, chunks = [], startedAt = 0, tick = null;
  let pending = null; // { blob, seconds, url }
  const playing = new Map(); // note id -> object URL

  const setStatus = (msg) => { $("#rec-status").textContent = msg; };
  const canRecord = () => Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);

  function cannotRecordMessage() {
    if (window.isSecureContext === false) {
      return "Recording needs a secure page: open the app at http://127.0.0.1:8000 or http://localhost (or over https). You can upload an audio file below instead.";
    }
    return "This browser cannot record audio. You can upload an audio file below instead.";
  }

  function micError(err) {
    const name = err && err.name;
    if (name === "NotAllowedError" || name === "SecurityError") {
      return "Microphone permission was denied. Allow the microphone for this page in the browser settings and press Record again, or upload an audio file below.";
    }
    if (name === "NotFoundError" || name === "OverconstrainedError") {
      return "No microphone was found on this device. You can upload an audio file below instead.";
    }
    if (name === "NotReadableError") return "The microphone is in use by another app. Close that app and try again.";
    return `The microphone could not start (${(err && err.message) || "unknown error"}). You can upload an audio file below instead.`;
  }

  function pickMime() {
    const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
    return types.find((t) => MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) || "";
  }

  /* ---------- Recording ---------- */

  async function start() {
    if (!canRecord()) { setStatus(cannotRecordMessage()); return; }
    discard();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setStatus(micError(err));
      return;
    }
    const mime = pickMime();
    try {
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    } catch (err) {
      stopStream();
      setStatus(`This browser cannot record audio here (${err.message}). You can upload an audio file below instead.`);
      return;
    }
    chunks = [];
    recorder.addEventListener("dataavailable", (e) => { if (e.data && e.data.size) chunks.push(e.data); });
    recorder.addEventListener("stop", onStopped);
    recorder.start(1000);
    startedAt = performance.now();
    $("#rec-timer").textContent = "0:00";
    $("#recorder").classList.add("recording");
    tick = setInterval(() => {
      const s = (performance.now() - startedAt) / 1000;
      $("#rec-timer").textContent = fmtDuration(s);
      if (s >= MAX_SECONDS) stop();
    }, 250);
    $("#rec-start").disabled = true;
    $("#rec-stop").disabled = false;
    $("#rec-stop").focus();
    setStatus("Recording. Press Stop when you are done.");
  }

  function stopStream() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  function stop() {
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }
  function onStopped() {
    clearInterval(tick);
    const seconds = (performance.now() - startedAt) / 1000;
    stopStream();
    $("#recorder").classList.remove("recording");
    $("#rec-start").disabled = false;
    $("#rec-stop").disabled = true;
    const type = (recorder && recorder.mimeType) || (chunks[0] && chunks[0].type) || "audio/webm";
    recorder = null;
    const blob = new Blob(chunks, { type });
    chunks = [];
    if (!blob.size) { setStatus("Nothing was recorded. Try again."); return; }
    pending = { blob, seconds, url: URL.createObjectURL(blob) };
    $("#rec-audio").src = pending.url;
    $("#rec-preview").hidden = false;
    setStatus(`Recorded ${fmtDuration(seconds)}. Listen, then save or discard.`);
    $("#rec-save").focus();
  }
  function discard() {
    if (pending) URL.revokeObjectURL(pending.url);
    pending = null;
    $("#rec-audio").removeAttribute("src");
    $("#rec-preview").hidden = true;
  }

  /** Duration of an audio blob in seconds, or null (some recorder output reports no duration). */
  function probeDuration(blob) {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(blob);
      const a = new Audio();
      const done = (v) => { URL.revokeObjectURL(url); resolve(Number.isFinite(v) && v > 0 ? v : null); };
      const t = setTimeout(() => done(null), 3000);
      a.preload = "metadata";
      a.onloadedmetadata = () => { clearTimeout(t); done(a.duration); };
      a.onerror = () => { clearTimeout(t); done(null); };
      a.src = url;
    });
  }

  async function store(blob, fileName, seconds) {
    const ext = blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "m4a" : "webm";
    return FilesDB.addVoiceNote(userId, {
      title: $("#rec-title").value.trim() || "Voice note",
      documentId: $("#rec-doc").value,
      fileName: fileName || `voice-note-${UI.today()}.${ext}`,
      duration: seconds,
    }, blob);
  }

  async function saveRecording() {
    if (!pending) return;
    const btn = $("#rec-save");
    btn.disabled = true;
    setStatus("Saving...");
    try {
      const n = await store(pending.blob, "", pending.seconds);
      discard();
      $("#rec-title").value = "";
      $("#rec-timer").textContent = "0:00";
      setStatus("");
      toast(`Saved "${n.title}".`);
      await refresh();
    } catch (err) {
      setStatus(err.code ? `Not saved: ${err.message}` : `Not saved. ${Documents.storageError(err)}`);
    } finally {
      btn.disabled = false;
    }
  }

  async function saveUpload() {
    const f = $("#rec-file").files[0];
    if (!f) { setStatus("Choose an audio file first."); $("#rec-file").focus(); return; }
    if (f.size > FilesDB.MAX_BYTES) { setStatus(`This file is ${fmtBytes(f.size)}. The limit is 20 MB.`); return; }
    const btn = $("#rec-file-save");
    btn.disabled = true;
    setStatus("Saving...");
    try {
      if (!$("#rec-title").value.trim()) $("#rec-title").value = f.name.replace(/\.[^.]+$/, "");
      const n = await store(f, f.name, await probeDuration(f));
      $("#rec-file").value = "";
      $("#rec-title").value = "";
      setStatus("");
      toast(`Saved "${n.title}".`);
      await refresh();
    } catch (err) {
      setStatus(err.code ? `Not saved: ${err.message}` : `Not saved. ${Documents.storageError(err)}`);
    } finally {
      btn.disabled = false;
    }
  }

  /* ---------- List ---------- */

  function docOptions(selected) {
    const docs = Documents.list();
    let html = '<option value="">No document</option>' + docs.map((d) =>
      `<option value="${esc(d.id)}"${d.id === selected ? " selected" : ""}>${esc(d.title)}</option>`).join("");
    if (selected && !docs.some((d) => d.id === selected)) html += `<option value="${esc(selected)}" selected>Document not loaded</option>`;
    return html;
  }
  function fillDocSelect() {
    const sel = $("#rec-doc");
    const docs = Documents.list();
    const want = preselect || sel.value;
    const keep = docs.some((d) => d.id === want) ? want : "";
    sel.innerHTML = '<option value="">No document</option>' + docs.map((d) =>
      `<option value="${esc(d.id)}"${d.id === keep ? " selected" : ""}>${esc(d.title)}</option>`).join("");
    if (keep) preselect = "";
  }

  async function refresh() {
    const forUser = userId;
    if (!forUser) return;
    try {
      const list = await FilesDB.listVoiceNotes(forUser);
      if (forUser !== userId) return;
      notes = list;
      loadError = null;
    } catch (err) {
      notes = [];
      loadError = err;
    }
    loaded = true;
    render();
    emit("files:changed", { kind: "voice" });
  }

  function releasePlayers() {
    playing.forEach((url) => URL.revokeObjectURL(url));
    playing.clear();
  }

  function render() {
    const list = $("#voice-list");
    releasePlayers();
    if (loadError) {
      $("#voice-count").textContent = "";
      list.innerHTML = `<li class="state error-state">${icon("voice")}<p>${esc(Documents.storageError(loadError))}</p></li>`;
      return;
    }
    if (!loaded) { list.innerHTML = '<li class="state"><p class="muted">Loading voice notes...</p></li>'; return; }
    const n = notes.length;
    $("#voice-count").textContent = n ? `${n} voice ${n === 1 ? "note" : "notes"}` : "";
    if (!n) {
      list.innerHTML = `<li class="state">${icon("voice")}<h3>No voice notes yet</h3>
        <p class="muted">Press Record to make the first one, or upload an audio file.</p></li>`;
      return;
    }
    list.innerHTML = notes.map((v) => {
      const id = esc(v.id);
      const doc = v.documentId ? Documents.get(v.documentId) : null;
      const meta = [fmtDate(v.createdAt), v.duration != null ? fmtDuration(v.duration) : "",
        `${FilesDB.AUDIO_TYPES[v.mime] || "Audio"}, ${fmtBytes(v.size)}`].filter(Boolean);
      const hidden = `<span class="visually-hidden"> ${esc(v.title)}</span>`;
      return `<li class="voice-card panel" data-id="${id}">
        <div class="voice-main">
          <button type="button" class="btn icon-only play-btn" data-act="play" aria-label="Play ${esc(v.title)}">${icon("play")}</button>
          <div class="voice-text">
            <h3 class="voice-title">${esc(v.title)}</h3>
            <p class="muted small">${meta.map(esc).join(" &middot; ")}</p>
            ${doc ? `<p class="small doc-link">${icon("link")}<a href="#documents" data-act="open-doc">${esc(doc.title)}</a></p>` : ""}
          </div>
        </div>
        <div class="player-slot"></div>
        <div class="card-actions">
          <button type="button" class="btn small" data-act="edit" aria-expanded="false">${icon("edit")}Rename${hidden}</button>
          <button type="button" class="btn small" data-act="download">${icon("download")}Download${hidden}</button>
          <button type="button" class="btn small danger" data-act="delete">${icon("trash")}Delete${hidden}</button>
        </div>
        <div class="edit-slot"></div>
      </li>`;
    }).join("");
  }

  async function onListClick(e) {
    const btn = e.target.closest("[data-act]");
    if (!btn) return;
    const card = btn.closest(".voice-card");
    const v = notes.find((x) => x.id === card.dataset.id);
    if (!v) return;
    const act = btn.dataset.act;
    if (act === "play") {
      const slot = card.querySelector(".player-slot");
      let audio = slot.querySelector("audio");
      if (!audio) {
        const blob = await FilesDB.getBlob(v.id).catch(() => null);
        if (!blob) { toast("The audio file is missing.", "error"); return; }
        const url = URL.createObjectURL(blob);
        playing.set(v.id, url);
        slot.innerHTML = `<audio controls src="${url}" aria-label="${esc(v.title)}"></audio>`;
        audio = slot.querySelector("audio");
      }
      audio.play().catch(() => { /* the user can press play on the controls */ });
      audio.focus();
    } else if (act === "download") {
      const blob = await FilesDB.getBlob(v.id).catch(() => null);
      if (blob) UI.download(blob, v.fileName); else toast("The audio file is missing.", "error");
    } else if (act === "open-doc") {
      e.preventDefault();
      location.hash = "#documents";
      const d = Documents.get(v.documentId);
      if (d) Documents.view(d);
    } else if (act === "edit") {
      const slot = card.querySelector(".edit-slot");
      const open = !slot.innerHTML;
      slot.innerHTML = open ? `<form class="form edit-form" novalidate aria-label="Rename ${esc(v.title)}">
          <div class="row">
            <label>Title<input name="title" maxlength="200" required value="${esc(v.title)}" /></label>
            <label>Attached to<select name="documentId">${docOptions(v.documentId)}</select></label>
          </div>
          <p class="form-status" role="status"></p>
          <div class="row-actions"><button type="submit" class="btn primary small">Save changes</button>
          <button type="button" class="btn small" data-act="cancel-edit">Cancel</button></div></form>` : "";
      btn.setAttribute("aria-expanded", String(open));
      if (open) slot.querySelector("input").focus();
    } else if (act === "cancel-edit") {
      card.querySelector(".edit-slot").innerHTML = "";
      const eb = card.querySelector('[data-act="edit"]');
      eb.setAttribute("aria-expanded", "false");
      eb.focus();
    } else if (act === "delete") {
      if (!confirm(`Delete the voice note "${v.title}"? This cannot be undone.`)) return;
      try {
        await FilesDB.deleteVoiceNote(v.id);
        toast(`Deleted "${v.title}".`);
        await refresh();
        $("#rec-title").focus();
      } catch (err) {
        toast(`Could not delete: ${err.message}`, "error");
      }
    }
  }

  async function onListSubmit(e) {
    e.preventDefault();
    const form = e.target;
    const card = form.closest(".voice-card");
    const title = form.elements.title.value.trim();
    const status = form.querySelector(".form-status");
    if (!title) { status.textContent = "The title cannot be empty."; form.elements.title.focus(); return; }
    try {
      await FilesDB.updateVoiceNote(card.dataset.id, { title, documentId: form.elements.documentId.value });
      toast("Changes saved.");
      await refresh();
    } catch (err) {
      status.textContent = `Not saved: ${err.message}`;
    }
  }

  function bind() {
    $("#rec-start").addEventListener("click", start);
    $("#rec-stop").addEventListener("click", stop);
    $("#rec-save").addEventListener("click", saveRecording);
    $("#rec-discard").addEventListener("click", () => { discard(); setStatus("Recording discarded."); $("#rec-start").focus(); });
    $("#rec-file-save").addEventListener("click", saveUpload);
    $("#voice-list").addEventListener("click", onListClick);
    $("#voice-list").addEventListener("submit", onListSubmit);

    document.addEventListener("user:open", (e) => {
      stop(); stopStream(); discard();
      userId = e.detail.id;
      notes = []; loaded = false; loadError = null; preselect = "";
      $("#rec-title").value = "";
      setStatus(canRecord() ? "" : cannotRecordMessage());
      render();
      refresh();
    });
    document.addEventListener("user:close", () => { stop(); stopStream(); discard(); userId = null; notes = []; releasePlayers(); });
    document.addEventListener("files:changed", (e) => {
      if (e.detail && e.detail.kind === "documents") {
        fillDocSelect();
        if (loaded) render();
      }
    });
    document.addEventListener("voice:for-document", (e) => {
      preselect = e.detail;
      fillDocSelect();
      setTimeout(() => $("#rec-title").focus(), 0);
    });
    window.addEventListener("pagehide", () => { stop(); stopStream(); });
  }

  bind();
  return { refresh, list: () => notes.slice() };
})();

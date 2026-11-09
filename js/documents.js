/* Health Diary: document library (add, list, search, view, download, edit, delete). Uses UI, FilesDB, App. */
const Documents = (() => {
  const { $, esc, icon, fmtDate, fmtBytes, toast, emit } = UI;

  const CATEGORIES = {
    prescription: "Prescription",
    lab_report: "Lab report",
    medical_certificate: "Medical certificate",
    discharge_summary: "Discharge summary",
    insurance: "Insurance paper",
    other: "Other",
  };
  const catText = (c) => CATEGORIES[c] || "Other";

  let userId = null;
  let docs = [];
  let loadError = null;
  let loaded = false;

  /* ---------- Shared form fields (upload, scan, edit) ---------- */

  function linkOptions(selected = "") {
    const u = App.user();
    if (!u) return '<option value="">None</option>';
    const visits = [...(u.visits || [])].sort((a, b) => (b.date || "").localeCompare(a.date || ""));
    const conds = [...(u.conditions || [])].sort((a, b) => a.name.localeCompare(b.name));
    const opt = (v, t) => `<option value="${esc(v)}"${v === selected ? " selected" : ""}>${esc(t)}</option>`;
    let html = opt("", "None");
    if (visits.length) {
      html += `<optgroup label="Visits &amp; reports">${visits.map((v) =>
        opt(`visit:${v.id}`, `${fmtDate(v.date)}, ${v.type}${v.doctor ? ", " + v.doctor : ""}`)).join("")}</optgroup>`;
    }
    if (conds.length) {
      html += `<optgroup label="Conditions">${conds.map((c) => opt(`condition:${c.id}`, `${c.name} (${c.status})`)).join("")}</optgroup>`;
    }
    // Keep a link to a record that was deleted visible rather than silently dropping it.
    if (selected && !html.includes(`value="${esc(selected)}"`)) html += opt(selected, "Removed record");
    return html;
  }

  /** Text for a document's link to a visit or condition, or "". */
  function linkText(link) {
    if (!link) return "";
    const [kind, id] = link.split(":");
    const u = App.user();
    if (!u) return "";
    if (kind === "visit") {
      const v = (u.visits || []).find((x) => x.id === id);
      return v ? `Visit: ${v.type}, ${fmtDate(v.date)}${v.doctor ? ", " + v.doctor : ""}` : "";
    }
    if (kind === "condition") {
      const c = (u.conditions || []).find((x) => x.id === id);
      return c ? `Condition: ${c.name}` : "";
    }
    return "";
  }

  /** HTML for the metadata fields of a document form. `p` is an id prefix, `d` the current values. */
  function fieldsHtml(p, d = {}) {
    const cats = Object.entries(CATEGORIES).map(([k, v]) =>
      `<option value="${k}"${k === (d.category || "other") ? " selected" : ""}>${esc(v)}</option>`).join("");
    return `
      <div class="row">
        <label for="${p}-title">Title<input id="${p}-title" name="title" maxlength="200" value="${esc(d.title || "")}" /></label>
        <label for="${p}-category">Category<select id="${p}-category" name="category">${cats}</select></label>
        <label for="${p}-date">Document date<input id="${p}-date" name="docDate" type="date" value="${esc(d.docDate || "")}" /></label>
      </div>
      <div class="row">
        <label for="${p}-tags">Tags (comma separated)<input id="${p}-tags" name="tags" maxlength="400" value="${esc((d.tags || []).join(", "))}" placeholder="e.g. blood test, cardiology" /></label>
        <label for="${p}-link">Linked visit or condition<select id="${p}-link" name="link">${linkOptions(d.link || "")}</select></label>
      </div>
      <label for="${p}-notes">Notes<textarea id="${p}-notes" name="notes" rows="2" maxlength="5000">${esc(d.notes || "")}</textarea></label>`;
  }

  function readFields(form) {
    const f = new FormData(form);
    return {
      title: String(f.get("title") || "").trim(),
      category: String(f.get("category") || "other"),
      docDate: String(f.get("docDate") || ""),
      tags: String(f.get("tags") || ""),
      link: String(f.get("link") || ""),
      notes: String(f.get("notes") || "").trim(),
    };
  }

  function storageError(err) {
    return `Files cannot be stored in this browser (${err && err.message ? err.message : err}). The rest of the diary still works.`;
  }

  /* ---------- Data ---------- */

  async function refresh() {
    const forUser = userId;
    if (!forUser) return;
    try {
      const list = await FilesDB.listDocuments(forUser);
      if (forUser !== userId) return; // the user changed while loading
      docs = list;
      loadError = null;
    } catch (err) {
      docs = [];
      loadError = err;
    }
    loaded = true;
    render();
    emit("files:changed", { kind: "documents" });
  }

  /** Store a new document for the current user and refresh the list. Returns the record. */
  async function add(meta, blob) {
    const d = await FilesDB.addDocument(userId, meta, blob);
    await refresh();
    return d;
  }

  /* ---------- Rendering ---------- */

  function matches(d, q, cat) {
    if (cat && d.category !== cat) return false;
    if (!q) return true;
    const hay = [d.title, d.notes, d.fileName, catText(d.category), ...(d.tags || [])].join(" ").toLowerCase();
    return q.toLowerCase().split(/\s+/).filter(Boolean).every((w) => hay.includes(w));
  }

  function thumbHtml(d) {
    if (d.thumb) return `<img src="${esc(d.thumb)}" alt="" loading="lazy" decoding="async" />`;
    return `<span class="thumb-label">${icon("documents")}<span>${esc(FilesDB.DOC_TYPES[d.mime] || "File")}</span></span>`;
  }

  function cardHtml(d) {
    const id = esc(d.id);
    const meta = [d.docDate ? `Dated ${fmtDate(d.docDate)}` : `Added ${fmtDate(d.createdAt)}`,
      `${FilesDB.DOC_TYPES[d.mime] || d.mime}, ${fmtBytes(d.size)}`];
    if (d.source === "scan") meta.push("Scanned");
    const link = linkText(d.link);
    const hidden = `<span class="visually-hidden"> ${esc(d.title)}</span>`;
    return `<article class="doc-card" data-id="${id}" aria-labelledby="doc-t-${id}">
      <button type="button" class="thumb" data-act="view" tabindex="-1" aria-hidden="true">${thumbHtml(d)}</button>
      <div class="doc-body">
        <span class="chip cat-${esc(d.category)}">${esc(catText(d.category))}</span>
        <h3 id="doc-t-${id}" class="doc-title">${esc(d.title)}</h3>
        <p class="muted small">${meta.map(esc).join(" &middot; ")}</p>
        ${link ? `<p class="small doc-link">${icon("link")}${esc(link)}</p>` : ""}
        ${d.tags && d.tags.length ? `<ul class="tags" aria-label="Tags">${d.tags.map((t) => `<li class="tag">${esc(t)}</li>`).join("")}</ul>` : ""}
        ${d.notes ? `<p class="doc-notes small">${esc(d.notes)}</p>` : ""}
        <div class="card-actions">
          <button type="button" class="btn small" data-act="view">${icon("eye")}View${hidden}</button>
          <button type="button" class="btn small" data-act="download">${icon("download")}Download${hidden}</button>
          <button type="button" class="btn small" data-act="edit" aria-expanded="false">${icon("edit")}Edit${hidden}</button>
          <button type="button" class="btn small" data-act="voice">${icon("voice")}Voice note${hidden}</button>
          <button type="button" class="btn small danger" data-act="delete">${icon("trash")}Delete${hidden}</button>
        </div>
        <div class="edit-slot"></div>
      </div>
    </article>`;
  }

  function render() {
    const list = $("#doc-list");
    if (!list) return;
    if (loadError) {
      $("#doc-count").textContent = "";
      list.innerHTML = `<div class="state error-state">${icon("documents")}<p>${esc(storageError(loadError))}</p></div>`;
      return;
    }
    if (!loaded) {
      list.innerHTML = `<div class="state"><p class="muted">Loading documents...</p></div>`;
      return;
    }
    const q = $("#doc-q").value.trim(), cat = $("#doc-cat").value;
    const shown = docs.filter((d) => matches(d, q, cat));
    const n = shown.length;
    $("#doc-count").textContent = docs.length
      ? (q || cat ? `${n} of ${docs.length} documents match` : `${n} ${n === 1 ? "document" : "documents"}`)
      : "";
    if (!docs.length) {
      list.innerHTML = `<div class="state">${icon("documents")}<h3>No documents yet</h3>
        <p class="muted">Add a PDF or a photo of a record, or scan a paper document with the camera.</p></div>`;
      return;
    }
    if (!n) {
      list.innerHTML = `<div class="state">${icon("search")}<h3>Nothing matches</h3>
        <p class="muted">Try a different word or choose All categories.</p></div>`;
      return;
    }
    list.innerHTML = shown.map(cardHtml).join("");
  }

  /* ---------- Actions ---------- */

  async function view(d) {
    try {
      const blob = await FilesDB.getBlob(d.id);
      if (!blob) throw new Error("the file is missing");
      UI.openViewer({ title: d.title, blob, mime: d.mime, fileName: d.fileName,
        meta: `${catText(d.category)} · ${d.docDate ? fmtDate(d.docDate) : "no date"} · ${fmtBytes(d.size)}` });
    } catch (err) {
      toast(`Could not open the document: ${err.message}`, "error");
    }
  }

  async function downloadDoc(d) {
    try {
      const blob = await FilesDB.getBlob(d.id);
      if (!blob) throw new Error("the file is missing");
      UI.download(blob, d.fileName);
    } catch (err) {
      toast(`Could not download: ${err.message}`, "error");
    }
  }

  function togglePanel(id, btnId, open) {
    const panel = $(id), btn = $(btnId);
    const show = open === undefined ? panel.hidden : open;
    panel.hidden = !show;
    btn.setAttribute("aria-expanded", String(show));
    if (show) panel.querySelector("h3").focus();
    return show;
  }

  function resetUpload() {
    const form = $("#doc-upload");
    form.reset();
    form.querySelector("[data-fields]").innerHTML = fieldsHtml("up");
    $("#doc-upload-status").textContent = "";
  }

  function bind() {
    $("#doc-cat").insertAdjacentHTML("beforeend",
      Object.entries(CATEGORIES).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join(""));

    $("#doc-add-btn").addEventListener("click", () => {
      if (!$("#doc-upload").hidden) { togglePanel("#doc-upload", "#doc-add-btn", false); return; }
      resetUpload();
      togglePanel("#doc-upload", "#doc-add-btn", true);
    });
    $("#doc-upload-cancel").addEventListener("click", () => {
      togglePanel("#doc-upload", "#doc-add-btn", false);
      $("#doc-add-btn").focus();
    });
    $("#doc-file").addEventListener("change", (e) => {
      const f = e.target.files[0];
      const title = $("#up-title");
      if (f && title && !title.value.trim()) title.value = f.name.replace(/\.[^.]+$/, "");
      $("#doc-upload-status").textContent = f && f.size > FilesDB.MAX_BYTES
        ? `This file is ${fmtBytes(f.size)}. The limit is 20 MB.` : "";
    });
    $("#doc-upload").addEventListener("submit", async (e) => {
      e.preventDefault();
      const status = $("#doc-upload-status");
      const file = $("#doc-file").files[0];
      if (!file) { status.textContent = "Choose a file first."; $("#doc-file").focus(); return; }
      if (file.size > FilesDB.MAX_BYTES) { status.textContent = `This file is ${fmtBytes(file.size)}. The limit is 20 MB.`; return; }
      const btn = $("#doc-upload-save");
      btn.disabled = true;
      status.textContent = "Saving...";
      try {
        const meta = readFields(e.target);
        meta.fileName = file.name;
        const type = await FilesDB.sniffDocument(file);
        if (type && type.startsWith("image/")) meta.thumb = await UI.thumbnail(file);
        const d = await add(meta, file);
        togglePanel("#doc-upload", "#doc-add-btn", false);
        toast(`Saved "${d.title}".`);
        $("#doc-add-btn").focus();
      } catch (err) {
        status.textContent = err.code ? `Not saved: ${err.message}` : `Not saved. ${storageError(err)}`;
      } finally {
        btn.disabled = false;
      }
    });

    let timer = null;
    $("#doc-q").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(render, 150); });
    $("#doc-cat").addEventListener("change", () => { clearTimeout(timer); render(); });

    $("#doc-list").addEventListener("click", async (e) => {
      const btn = e.target.closest("[data-act]");
      if (!btn) return;
      const card = btn.closest(".doc-card");
      const d = docs.find((x) => x.id === card.dataset.id);
      if (!d) return;
      const act = btn.dataset.act;
      if (act === "view") view(d);
      else if (act === "download") downloadDoc(d);
      else if (act === "voice") {
        emit("voice:for-document", d.id);
        location.hash = "#voice";
      } else if (act === "edit") {
        const slot = card.querySelector(".edit-slot");
        const open = !slot.innerHTML;
        slot.innerHTML = open ? `<form class="form edit-form" novalidate aria-label="Edit ${esc(d.title)}">
            ${fieldsHtml(`e-${d.id}`, d)}
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
        if (!confirm(`Delete "${d.title}"? This cannot be undone. Voice notes attached to it are kept.`)) return;
        try {
          await FilesDB.deleteDocument(d.id);
          toast(`Deleted "${d.title}".`);
          await refresh();
          if (typeof Voice !== "undefined") Voice.refresh();
          $("#doc-q").focus();
        } catch (err) {
          toast(`Could not delete: ${err.message}`, "error");
        }
      }
    });

    $("#doc-list").addEventListener("submit", async (e) => {
      e.preventDefault();
      const form = e.target;
      const card = form.closest(".doc-card");
      const meta = readFields(form);
      const status = form.querySelector(".form-status");
      if (!meta.title) { status.textContent = "The title cannot be empty."; form.elements.title.focus(); return; }
      try {
        await FilesDB.updateDocument(card.dataset.id, meta);
        toast("Changes saved.");
        await refresh();
        const again = $(`.doc-card[data-id="${CSS.escape(card.dataset.id)}"] [data-act="edit"]`);
        if (again) again.focus();
      } catch (err) {
        status.textContent = `Not saved: ${err.message}`;
      }
    });

    document.addEventListener("user:open", (e) => {
      userId = e.detail.id;
      docs = []; loaded = false; loadError = null;
      $("#doc-q").value = ""; $("#doc-cat").value = "";
      $("#doc-upload").hidden = true;
      $("#doc-add-btn").setAttribute("aria-expanded", "false");
      render();
      refresh();
    });
    document.addEventListener("user:close", () => { userId = null; docs = []; loaded = false; });
    // Visits or conditions changed: link texts and options may change.
    document.addEventListener("user:data", () => { if (loaded) render(); });
  }

  bind();

  return {
    CATEGORIES, catText, fieldsHtml, readFields, linkText, storageError,
    add, refresh, view,
    list: () => docs.slice(),
    get: (id) => docs.find((d) => d.id === id) || null,
    isLoaded: () => loaded,
    error: () => loadError,
  };
})();

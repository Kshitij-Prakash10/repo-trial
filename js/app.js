/* Health Diary: UI logic (users, navigation, dashboard, record tables, profile). */
const App = (() => {
  const { $, $$, esc, fmtDate, fmtBytes, today, icon, toast, emit } = UI;
  const byDateDesc = (key) => (a, b) => (b[key] || "").localeCompare(a[key] || "");
  const VIEWS = ["dashboard", "vitals", "conditions", "medications", "allergies", "visits", "documents", "scan", "voice", "profile"];
  const LISTS = ["vitals", "conditions", "medications", "allergies", "visits"];

  let user = null;

  /* ---------- Table definitions for each record type ---------- */
  const TABLES = {
    vitals: {
      sortKey: "date",
      label: (r) => `vitals entry of ${fmtDate(r.date)}`,
      empty: "No vitals yet. Add your first entry above.",
      columns: [
        ["Date", (r) => fmtDate(r.date)],
        ["Height", (r) => (r.height ? r.height + " cm" : "—")],
        ["Weight", (r) => (r.weight ? r.weight + " kg" : "—")],
        ["BMI", (r) => bmi(r.height, r.weight) ?? "—"],
        ["BP", (r) => r.bp || "—"],
        ["HR", (r) => (r.heartRate ? r.heartRate + " bpm" : "—")],
        ["Sugar", (r) => (r.sugar ? r.sugar + " mg/dL" : "—")],
        ["Notes", (r) => r.notes],
      ],
    },
    conditions: {
      sortKey: "diagnosed",
      label: (r) => `condition ${r.name}`,
      empty: "No conditions recorded.",
      columns: [
        ["Condition", (r) => r.name],
        ["Diagnosed", (r) => fmtDate(r.diagnosed)],
        ["Status", (r) => badge(r.status, r.status === "Resolved" ? "resolved" : r.status === "Chronic" ? "chronic" : "")],
        ["Notes", (r) => r.notes],
      ],
    },
    medications: {
      sortKey: "start",
      label: (r) => `medication ${r.name}`,
      empty: "No medications recorded.",
      columns: [
        ["Medicine", (r) => r.name],
        ["Dosage", (r) => r.dosage],
        ["From", (r) => fmtDate(r.start)],
        ["To", (r) => (r.end ? fmtDate(r.end) : "Ongoing")],
        ["Status", (r) => (isCurrentMed(r) ? badge("Current") : badge("Past", "past"))],
        ["For", (r) => r.reason],
      ],
    },
    allergies: {
      sortKey: "name",
      label: (r) => `allergy ${r.name}`,
      empty: "No allergies recorded.",
      columns: [
        ["Allergen", (r) => r.name],
        ["Severity", (r) => badge(r.severity, r.severity === "Severe" ? "severe" : r.severity === "Moderate" ? "moderate" : "")],
        ["Reaction", (r) => r.reaction],
      ],
    },
    visits: {
      sortKey: "date",
      label: (r) => `${r.type} of ${fmtDate(r.date)}`,
      empty: "No visits or reports yet.",
      columns: [
        ["Date", (r) => fmtDate(r.date)],
        ["Type", (r) => r.type],
        ["Doctor / Hospital", (r) => r.doctor],
        ["Summary", (r) => r.notes],
      ],
    },
  };

  /* ---------- Helpers ---------- */
  function bmi(h, w) {
    if (!h || !w) return null;
    const m = h / 100;
    return (w / (m * m)).toFixed(1);
  }
  function badge(text, cls = "") {
    return { html: `<span class="badge ${cls}">${esc(text)}</span>` };
  }
  function isCurrentMed(m) {
    return !m.end || m.end >= today();
  }
  function age(dob) {
    if (!dob) return null;
    const d = new Date(dob), n = new Date();
    let a = n.getFullYear() - d.getFullYear();
    if (n < new Date(n.getFullYear(), d.getMonth(), d.getDate())) a--;
    return a;
  }
  function cell(v) {
    if (v && typeof v === "object" && "html" in v) return v.html;
    return esc(v || "");
  }
  function save() { Storage.saveUser(user); }
  const recordCount = (u) => LISTS.reduce((s, k) => s + (Array.isArray(u[k]) ? u[k].length : 0), 0);
  // Tolerate older or hand-edited saved users with a missing list (in memory only; adds, never changes).
  function normalise(u) {
    LISTS.forEach((k) => { if (!Array.isArray(u[k])) u[k] = []; });
    u.profile = u.profile || { name: "Unnamed" };
    return u;
  }

  /* ---------- Auth / user selection ---------- */
  function renderUserList() {
    const list = $("#user-list");
    const users = Storage.listUsers();
    list.innerHTML = users.length
      ? users.map((u) => `
          <li><button type="button" class="user-btn" data-user="${esc(u.id)}">
            <span class="avatar small-avatar" aria-hidden="true">${esc((u.profile.name || "?").trim().charAt(0).toUpperCase())}</span>
            <span class="user-btn-name">${esc(u.profile.name)}</span>
            <span class="muted small" data-usage="${esc(u.id)}">${recordCount(u)} records</span>
          </button></li>`).join("")
      : `<li class="empty">No users yet. Create one below.</li>`;
    users.forEach(async (u) => {
      try {
        const use = await FilesDB.usage(u.id);
        const n = use.documents.count + use.voiceNotes.count;
        const el = $(`[data-usage="${CSS.escape(u.id)}"]`);
        if (el && n) el.textContent = `${recordCount(u)} records, ${n} ${n === 1 ? "file" : "files"} (${fmtBytes(use.bytes)})`;
      } catch (err) { /* file storage not available: keep the record count */ }
    });
  }

  function openUser(id) {
    const u = Storage.getUser(id);
    if (!u) return showAuth();
    user = normalise(u);
    Storage.setCurrentUserId(id);
    $("#auth-screen").classList.add("hidden");
    $("#app").classList.remove("hidden");
    renderAll();
    emit("user:open", user);
    showView(viewFromHash() || "dashboard");
  }

  function showAuth() {
    if (user) emit("user:close");
    user = null;
    Storage.setCurrentUserId(null);
    closeMenu();
    $("#app").classList.add("hidden");
    $("#auth-screen").classList.remove("hidden");
    if (location.hash) history.replaceState(null, "", location.pathname + location.search);
    document.title = "Health Diary";
    renderUserList();
  }

  /* ---------- Navigation ---------- */
  function viewFromHash() {
    const h = location.hash.slice(1);
    return VIEWS.includes(h) ? h : null;
  }

  function showView(name, focus = false) {
    $$(".view").forEach((s) => s.classList.toggle("hidden", s.dataset.section !== name));
    $$(".sidebar nav a").forEach((a) => {
      const on = a.dataset.view === name;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    const title = $(`[data-section="${name}"] h2`);
    document.title = `${title ? title.textContent : "Health Diary"} - Health Diary`;
    closeMenu();
    if (focus && title) title.focus();
    emit("view:show", { name });
  }

  function closeMenu() {
    $("#app").classList.remove("menu-open");
    $("#menu-btn").setAttribute("aria-expanded", "false");
  }

  /* ---------- Rendering ---------- */
  function renderAll() {
    renderHeader();
    renderDashboard();
    Object.keys(TABLES).forEach(renderTable);
    renderProfileForm();
    renderStorage();
  }

  function renderHeader() {
    const p = user.profile;
    $("#user-name").textContent = p.name;
    $("#user-avatar").textContent = (p.name || "?").trim().charAt(0).toUpperCase();
    const a = age(p.dob);
    $("#user-meta").textContent = [a != null ? `${a} yrs` : null, p.gender, p.bloodGroup]
      .filter(Boolean).join(" · ");
  }

  function latest(field) {
    const entry = [...user.vitals].sort(byDateDesc("date")).find((v) => v[field]);
    return entry ? { value: entry[field], date: entry.date } : null;
  }

  function renderDashboard() {
    const h = latest("height"), w = latest("weight"), bp = latest("bp"), hr = latest("heartRate");
    const cards = [
      ["Height", h && `${h.value} cm`, h?.date],
      ["Weight", w && `${w.value} kg`, w?.date],
      ["BMI", bmi(h?.value, w?.value), null],
      ["Blood pressure", bp?.value, bp?.date],
      ["Heart rate", hr && `${hr.value} bpm`, hr?.date],
      ["Blood group", user.profile.bloodGroup, null],
    ];
    $("#current-cards").innerHTML = cards.map(([label, value, date]) => `
      <div class="card${value ? "" : " card-empty"}">
        <div class="label">${label}</div>
        <div class="value">${esc(value || "—")}</div>
        <div class="sub">${date ? "as of " + fmtDate(date) : value ? "&nbsp;" : "Not recorded yet"}</div>
      </div>`).join("");

    const active = user.conditions.filter((c) => c.status !== "Resolved");
    $("#dash-conditions").innerHTML = active.length
      ? active.map((c) => `<li><span>${esc(c.name)}</span> ${cell(badge(c.status, c.status === "Chronic" ? "chronic" : ""))}</li>`).join("")
      : `<li class="muted">No active conditions</li>`;

    const meds = user.medications.filter(isCurrentMed);
    $("#dash-medications").innerHTML = meds.length
      ? meds.map((m) => `<li><strong>${esc(m.name)}</strong> <span class="muted">${esc(m.dosage)}</span></li>`).join("")
      : `<li class="muted">No current medications</li>`;

    // Combined timeline of past records
    const events = [
      ...user.vitals.map((v) => ({ date: v.date, text: `Vitals recorded${v.weight ? `, ${v.weight} kg` : ""}` })),
      ...user.conditions.filter((c) => c.diagnosed).map((c) => ({ date: c.diagnosed, text: `Diagnosed: ${c.name}` })),
      ...user.medications.filter((m) => m.start).map((m) => ({ date: m.start, text: `Started ${m.name}` })),
      ...user.visits.map((v) => ({ date: v.date, text: `${v.type}${v.doctor ? " at " + v.doctor : ""}` })),
    ].sort(byDateDesc("date")).slice(0, 10);

    $("#timeline").innerHTML = events.length
      ? events.map((e) => `<li><div class="date">${fmtDate(e.date)}</div>${esc(e.text)}</li>`).join("")
      : `<li class="muted">No records yet. Start by adding your vitals.</li>`;

    renderDashDocuments();
  }

  function renderDashDocuments() {
    const box = $("#dash-documents");
    if (Documents.error()) {
      box.innerHTML = `<li class="muted">${esc(Documents.storageError(Documents.error()))}</li>`;
      return;
    }
    if (!Documents.isLoaded()) { box.innerHTML = `<li class="muted">Loading...</li>`; return; }
    const docs = Documents.list().slice(0, 5);
    box.innerHTML = docs.length
      ? docs.map((d) => `<li>
          <span class="mini-thumb" aria-hidden="true">${d.thumb ? `<img src="${esc(d.thumb)}" alt="" />` : icon("documents")}</span>
          <span class="mini-text"><button type="button" class="linklike" data-open-doc="${esc(d.id)}">${esc(d.title)}</button>
          <span class="muted small">${esc(Documents.catText(d.category))} &middot; ${esc(d.docDate ? fmtDate(d.docDate) : fmtDate(d.createdAt))}</span></span>
        </li>`).join("")
      : `<li class="muted">No documents yet. <a href="#documents">Add one</a> or <a href="#scan">scan one</a>.</li>`;
  }

  function renderTable(type) {
    const def = TABLES[type];
    const rows = [...user[type]].sort(
      type === "allergies" ? (a, b) => (a.name || "").localeCompare(b.name || "") : byDateDesc(def.sortKey));
    const table = $(`[data-table="${type}"]`);
    const caption = `<caption class="visually-hidden">${esc($(`[data-section="${type}"] h2`).textContent)}</caption>`;
    if (!rows.length) {
      table.innerHTML = `${caption}<tbody><tr class="empty-row"><td>${esc(def.empty)}</td></tr></tbody>`;
      return;
    }
    table.innerHTML = `${caption}
      <thead><tr>${def.columns.map(([h]) => `<th scope="col">${h}</th>`).join("")}<th scope="col"><span class="visually-hidden">Actions</span></th></tr></thead>
      <tbody>${rows.map((r) => `
        <tr>${def.columns.map(([h, fn]) => `<td data-label="${esc(h)}">${cell(fn(r))}</td>`).join("")}
          <td class="actions-cell"><button type="button" class="btn link" data-delete="${type}:${esc(r.id)}"
            aria-label="Delete ${esc(def.label(r))}">${icon("trash")}Delete</button></td>
        </tr>`).join("")}
      </tbody>`;
  }

  function renderProfileForm() {
    const f = $("#profile-form");
    Object.entries(user.profile).forEach(([k, v]) => { if (f.elements[k]) f.elements[k].value = v || ""; });
  }

  let storageSeq = 0;
  async function renderStorage() {
    if (!user) return;
    const mine = ++storageSeq;
    const forUser = user.id;
    let use = null, est = null, err = null;
    try {
      [use, est] = await Promise.all([FilesDB.usage(forUser), FilesDB.browserEstimate()]);
    } catch (e) { err = e; }
    if (mine !== storageSeq || !user || user.id !== forUser) return;
    if (err) {
      $$(".js-storage-used").forEach((el) => { el.textContent = "File storage not available"; });
      $("#storage-facts").innerHTML = `<div><dt>Files</dt><dd>${esc(Documents.storageError(err))}</dd></div>`;
      return;
    }
    const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
    $$(".js-storage-used").forEach((el) => {
      el.textContent = `Files: ${fmtBytes(use.bytes)} (${plural(use.documents.count, "document", "documents")}, ${plural(use.voiceNotes.count, "voice note", "voice notes")})`;
    });
    $("#storage-facts").innerHTML = `
      <div><dt>Documents</dt><dd>${plural(use.documents.count, "file", "files")}, ${fmtBytes(use.documents.bytes)}</dd></div>
      <div><dt>Voice notes</dt><dd>${plural(use.voiceNotes.count, "note", "notes")}, ${fmtBytes(use.voiceNotes.bytes)}</dd></div>
      <div><dt>Total for ${esc(user.profile.name)}</dt><dd>${fmtBytes(use.bytes)}</dd></div>
      ${est && est.quota ? `<div><dt>This browser, all users</dt><dd>${fmtBytes(est.usage || 0)} used of about ${fmtBytes(est.quota)} available (browser estimate)</dd></div>` : ""}`;
  }

  /* ---------- Events ---------- */
  function formData(form) {
    return Object.fromEntries(new FormData(form).entries());
  }

  function bindEvents() {
    $("#new-user-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const data = formData(e.target);
      data.name = String(data.name || "").trim();
      if (!data.name) { e.target.elements.name.focus(); return; }
      const u = Storage.createUser(data);
      e.target.reset();
      if (location.hash) history.replaceState(null, "", location.pathname + location.search);
      openUser(u.id);
      toast(`Welcome, ${u.profile.name}. Your diary is ready.`);
    });

    $("#user-list").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-user]");
      if (btn) openUser(btn.dataset.user);
    });

    $("#logout-btn").addEventListener("click", showAuth);

    $("#menu-btn").addEventListener("click", () => {
      const open = !$("#app").classList.contains("menu-open");
      $("#app").classList.toggle("menu-open", open);
      $("#menu-btn").setAttribute("aria-expanded", String(open));
      if (open) ($(".sidebar nav a.active") || $(".sidebar nav a")).focus();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && $("#app").classList.contains("menu-open")) {
        closeMenu();
        $("#menu-btn").focus();
      }
    });

    window.addEventListener("hashchange", () => {
      const v = viewFromHash();
      if (v && user) showView(v, true);
    });
    // Clicking the link of the view that is already open fires no hashchange, so handle it here.
    $$(".sidebar nav a").forEach((a) => a.addEventListener("click", () => {
      if (location.hash === `#${a.dataset.view}`) showView(a.dataset.view, true);
    }));

    // Generic "add record" forms
    $$("[data-form]").forEach((form) => {
      const dateInput = form.querySelector('input[name="date"]');
      if (dateInput) dateInput.value = today();
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        const type = form.dataset.form;
        const data = formData(form);
        if (type === "vitals" && !["height", "weight", "bp", "heartRate", "sugar", "notes"].some((k) => String(data[k] || "").trim())) {
          toast("Enter at least one value.", "error");
          form.elements.height.focus();
          return;
        }
        user[type].push({ id: Storage.uid(), ...data });
        save();
        form.reset();
        if (dateInput) dateInput.value = today();
        renderTable(type);
        renderDashboard();
        if (type === "visits" || type === "conditions") emit("user:data", user);
        toast("Saved.");
      });
    });

    // Delete buttons in tables
    $(".content").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-delete]");
      if (!btn) return;
      const [type, id] = btn.dataset.delete.split(":");
      if (!confirm("Delete this entry?")) return;
      user[type] = user[type].filter((r) => r.id !== id);
      save();
      renderTable(type);
      renderDashboard();
      if (type === "visits" || type === "conditions") emit("user:data", user);
      toast("Entry deleted.");
      $(`[data-section="${type}"] h2`).focus();
    });

    // Dashboard: open a document from the "Latest documents" list
    $("#dash-documents").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-open-doc]");
      if (!btn) return;
      const d = Documents.get(btn.dataset.openDoc);
      if (d) Documents.view(d);
    });

    $("#profile-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const data = formData(e.target);
      data.name = String(data.name || "").trim() || user.profile.name;
      user.profile = { ...user.profile, ...data };
      save();
      renderHeader();
      renderDashboard();
      renderStorage();
      toast("Profile saved.");
    });

    $("#delete-user").addEventListener("click", async () => {
      if (!confirm(`Delete ${user.profile.name} and all their records, documents and voice notes?`)) return;
      const id = user.id, name = user.profile.name;
      try {
        await FilesDB.deleteUserFiles(id);
      } catch (err) {
        // No file storage in this browser means there are no files to delete.
        console.warn("Could not delete the user's files:", err);
      }
      Storage.deleteUser(id);
      showAuth();
      toast(`Deleted ${name}.`);
    });

    document.addEventListener("files:changed", () => {
      if (!user) return;
      renderDashDocuments();
      renderStorage();
    });
  }

  function boot() {
    bindEvents();
    const current = Storage.getCurrentUserId();
    if (current && Storage.getUser(current)) openUser(current); else showAuth();
  }

  return { user: () => user, showView, boot };
})();

App.boot();

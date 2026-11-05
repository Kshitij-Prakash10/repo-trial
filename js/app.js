/* Health Diary – UI logic */
(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
  const fmtDate = (d) => (d ? new Date(d + "T00:00:00").toLocaleDateString() : "—");
  const byDateDesc = (key) => (a, b) => (b[key] || "").localeCompare(a[key] || "");
  const today = () => new Date().toISOString().slice(0, 10);

  let user = null;

  /* ---------- Table definitions for each record type ---------- */
  const TABLES = {
    vitals: {
      sortKey: "date",
      columns: [
        ["Date", (r) => fmtDate(r.date)],
        ["Height", (r) => (r.height ? r.height + " cm" : "—")],
        ["Weight", (r) => (r.weight ? r.weight + " kg" : "—")],
        ["BMI", (r) => bmi(r.height, r.weight) ?? "—"],
        ["BP", (r) => r.bp || "—"],
        ["HR", (r) => r.heartRate || "—"],
        ["Sugar", (r) => r.sugar || "—"],
        ["Notes", (r) => r.notes],
      ],
    },
    conditions: {
      sortKey: "diagnosed",
      columns: [
        ["Condition", (r) => r.name],
        ["Diagnosed", (r) => fmtDate(r.diagnosed)],
        ["Status", (r) => badge(r.status, r.status === "Resolved" ? "resolved" : "")],
        ["Notes", (r) => r.notes],
      ],
    },
    medications: {
      sortKey: "start",
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
      columns: [
        ["Allergen", (r) => r.name],
        ["Severity", (r) => badge(r.severity, r.severity === "Severe" ? "severe" : "")],
        ["Reaction", (r) => r.reaction],
      ],
    },
    visits: {
      sortKey: "date",
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

  /* ---------- Auth / user selection ---------- */
  function renderUserList() {
    const list = $("#user-list");
    const users = Storage.listUsers();
    list.innerHTML = users.length
      ? users.map((u) => `
          <li><button class="btn" data-user="${u.id}">
            <span>${esc(u.profile.name)}</span>
            <span class="muted small">${u.vitals.length + u.visits.length} records</span>
          </button></li>`).join("")
      : `<li class="empty">No users yet — create one below.</li>`;
  }

  function openUser(id) {
    user = Storage.getUser(id);
    if (!user) return showAuth();
    Storage.setCurrentUserId(id);
    $("#auth-screen").classList.add("hidden");
    $("#app").classList.remove("hidden");
    renderAll();
    showView("dashboard");
  }

  function showAuth() {
    user = null;
    Storage.setCurrentUserId(null);
    $("#app").classList.add("hidden");
    $("#auth-screen").classList.remove("hidden");
    renderUserList();
  }

  /* ---------- Navigation ---------- */
  function showView(name) {
    $$(".view").forEach((s) => s.classList.toggle("hidden", s.dataset.section !== name));
    $$(".sidebar nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === name));
  }

  /* ---------- Rendering ---------- */
  function renderAll() {
    renderHeader();
    renderDashboard();
    Object.keys(TABLES).forEach(renderTable);
    renderProfileForm();
  }

  function renderHeader() {
    const p = user.profile;
    $("#user-name").textContent = p.name;
    $("#user-avatar").textContent = p.name.trim().charAt(0).toUpperCase();
    const a = age(p.dob);
    $("#user-meta").textContent = [a != null ? `${a} yrs` : null, p.gender, p.bloodGroup]
      .filter(Boolean).join(" · ");
  }

  function latest(field) {
    const entry = [...user.vitals].sort(byDateDesc("date")).find((v) => v[field]);
    return entry ? { value: entry[field], date: entry.date } : null;
  }

  function renderDashboard() {
    const h = latest("height"), w = latest("weight");
    const cards = [
      ["Height", h && `${h.value} cm`, h?.date],
      ["Weight", w && `${w.value} kg`, w?.date],
      ["BMI", bmi(h?.value, w?.value), null],
      ["Blood pressure", latest("bp")?.value, latest("bp")?.date],
      ["Heart rate", latest("heartRate") && `${latest("heartRate").value} bpm`, latest("heartRate")?.date],
      ["Blood group", user.profile.bloodGroup, null],
    ];
    $("#current-cards").innerHTML = cards.map(([label, value, date]) => `
      <div class="card">
        <div class="label">${label}</div>
        <div class="value">${esc(value || "—")}</div>
        <div class="sub">${date ? "as of " + fmtDate(date) : "&nbsp;"}</div>
      </div>`).join("");

    const active = user.conditions.filter((c) => c.status !== "Resolved");
    $("#dash-conditions").innerHTML = active.length
      ? active.map((c) => `<li>${esc(c.name)} ${cell(badge(c.status))}</li>`).join("")
      : `<li class="muted">No active conditions</li>`;

    const meds = user.medications.filter(isCurrentMed);
    $("#dash-medications").innerHTML = meds.length
      ? meds.map((m) => `<li><strong>${esc(m.name)}</strong> <span class="muted">${esc(m.dosage)}</span></li>`).join("")
      : `<li class="muted">No current medications</li>`;

    // Combined timeline of past records
    const events = [
      ...user.vitals.map((v) => ({ date: v.date, text: `Vitals recorded${v.weight ? ` — ${v.weight} kg` : ""}` })),
      ...user.conditions.filter((c) => c.diagnosed).map((c) => ({ date: c.diagnosed, text: `Diagnosed: ${c.name}` })),
      ...user.medications.filter((m) => m.start).map((m) => ({ date: m.start, text: `Started ${m.name}` })),
      ...user.visits.map((v) => ({ date: v.date, text: `${v.type}${v.doctor ? " at " + v.doctor : ""}` })),
    ].sort(byDateDesc("date")).slice(0, 10);

    $("#timeline").innerHTML = events.length
      ? events.map((e) => `<li><div class="date">${fmtDate(e.date)}</div>${esc(e.text)}</li>`).join("")
      : `<li class="muted">No records yet. Start by adding your vitals.</li>`;
  }

  function renderTable(type) {
    const def = TABLES[type];
    const rows = [...user[type]].sort(
      type === "allergies" ? (a, b) => a.name.localeCompare(b.name) : byDateDesc(def.sortKey));
    const table = $(`[data-table="${type}"]`);
    if (!rows.length) {
      table.innerHTML = `<tr><td class="muted">No entries yet.</td></tr>`;
      return;
    }
    table.innerHTML = `
      <thead><tr>${def.columns.map(([h]) => `<th>${h}</th>`).join("")}<th></th></tr></thead>
      <tbody>${rows.map((r) => `
        <tr>${def.columns.map(([, fn]) => `<td>${cell(fn(r))}</td>`).join("")}
          <td><button class="btn link" data-delete="${type}:${r.id}">Delete</button></td>
        </tr>`).join("")}
      </tbody>`;
  }

  function renderProfileForm() {
    const f = $("#profile-form");
    Object.entries(user.profile).forEach(([k, v]) => { if (f.elements[k]) f.elements[k].value = v || ""; });
  }

  /* ---------- Events ---------- */
  function formData(form) {
    return Object.fromEntries(new FormData(form).entries());
  }

  function bindEvents() {
    $("#new-user-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const u = Storage.createUser(formData(e.target));
      e.target.reset();
      openUser(u.id);
    });

    $("#user-list").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-user]");
      if (btn) openUser(btn.dataset.user);
    });

    $("#logout-btn").addEventListener("click", showAuth);

    $$(".sidebar nav a").forEach((a) => a.addEventListener("click", (e) => {
      e.preventDefault();
      showView(a.dataset.view);
    }));

    // Generic "add record" forms
    $$("[data-form]").forEach((form) => {
      const dateInput = form.querySelector('input[name="date"]');
      if (dateInput) dateInput.value = today();
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        const type = form.dataset.form;
        user[type].push({ id: Storage.uid(), ...formData(form) });
        save();
        form.reset();
        if (dateInput) dateInput.value = today();
        renderTable(type);
        renderDashboard();
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
    });

    $("#profile-form").addEventListener("submit", (e) => {
      e.preventDefault();
      user.profile = { ...user.profile, ...formData(e.target) };
      save();
      renderHeader();
      renderDashboard();
      alert("Profile saved.");
    });

    $("#delete-user").addEventListener("click", () => {
      if (!confirm(`Delete ${user.profile.name} and all their records?`)) return;
      Storage.deleteUser(user.id);
      showAuth();
    });
  }

  /* ---------- Boot ---------- */
  bindEvents();
  const current = Storage.getCurrentUserId();
  current && Storage.getUser(current) ? openUser(current) : showAuth();
})();

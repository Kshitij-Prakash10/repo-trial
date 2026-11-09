/* Health Diary: small shared UI helpers (DOM, formatting, icons, toasts, file viewer). */
const UI = (() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

  /** Local date as YYYY-MM-DD. */
  const today = () => {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  };

  /** Format a YYYY-MM-DD or ISO 8601 string for display; "—" when empty. */
  function fmtDate(v) {
    if (!v) return "—";
    const d = new Date(v.length === 10 ? v + "T00:00:00" : v);
    return Number.isNaN(d.getTime()) ? String(v)
      : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  }

  /** Bytes as "512 B", "34 kB", "2.1 MB". */
  function fmtBytes(n) {
    if (!n) return "0 B";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${Math.round(n / 1024)} kB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1048576).toFixed(1)} MB`;
    return `${(n / 1073741824).toFixed(1)} GB`;
  }

  /** Seconds as m:ss. */
  function fmtDuration(s) {
    if (s == null || !Number.isFinite(s)) return "";
    s = Math.max(0, Math.round(s));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  /** Inline SVG icon from the sprite in index.html (decorative). */
  const icon = (name, cls = "") =>
    `<svg class="icon ${cls}" aria-hidden="true" focusable="false"><use href="#i-${name}"></use></svg>`;

  /** Show a short message in the polite live region. kind: "" | "error". */
  function toast(msg, kind = "") {
    const box = $("#toasts");
    if (!box) return;
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    if (kind === "error") el.setAttribute("role", "alert");
    el.textContent = msg;
    box.appendChild(el);
    while (box.children.length > 3) box.firstElementChild.remove(); // keep the stack short
    setTimeout(() => { el.classList.add("leaving"); setTimeout(() => el.remove(), 300); }, kind === "error" ? 6000 : 3500);
  }

  /** A small JPEG data: URL thumbnail of an image blob (longest side `max` px), or "" if it cannot be decoded. */
  async function thumbnail(blob, max = 240) {
    try {
      let src, w, h, revoke = null;
      if (window.createImageBitmap) {
        src = await createImageBitmap(blob);
        w = src.width; h = src.height;
      } else {
        const url = URL.createObjectURL(blob);
        revoke = url;
        src = await new Promise((res, rej) => {
          const im = new Image();
          im.onload = () => res(im); im.onerror = rej; im.src = url;
        });
        w = src.naturalWidth; h = src.naturalHeight;
      }
      const s = Math.min(1, max / Math.max(w, h));
      const cv = document.createElement("canvas");
      cv.width = Math.max(1, Math.round(w * s)); cv.height = Math.max(1, Math.round(h * s));
      const ctx = cv.getContext("2d");
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, cv.width, cv.height);
      ctx.drawImage(src, 0, 0, cv.width, cv.height);
      if (src.close) src.close();
      if (revoke) URL.revokeObjectURL(revoke);
      return cv.toDataURL("image/jpeg", 0.72);
    } catch (err) {
      return "";
    }
  }

  /** Save a blob to the user's downloads under `fileName`. */
  function download(blob, fileName) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = fileName || "file";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }

  /* ---------- File viewer (a modal <dialog>) ---------- */
  let viewerUrl = null;
  function closeViewer() {
    const dlg = $("#viewer");
    if (dlg.open) dlg.close();
  }
  /** Show an image or PDF blob in the viewer dialog. */
  function openViewer({ title, blob, mime, fileName, meta = "" }) {
    const dlg = $("#viewer");
    if (viewerUrl) URL.revokeObjectURL(viewerUrl);
    viewerUrl = URL.createObjectURL(blob);
    $("#viewer-title").textContent = title;
    $("#viewer-meta").textContent = meta;
    const body = $("#viewer-body");
    if (mime.startsWith("image/")) {
      body.innerHTML = `<img src="${viewerUrl}" alt="${esc(title)}">`;
    } else {
      body.innerHTML = `<iframe src="${viewerUrl}" title="${esc(title)} (PDF)"></iframe>
        <p class="muted small">If the PDF does not show here, use Open in new tab or Download.</p>`;
    }
    $("#viewer-open").href = viewerUrl;
    $("#viewer-download").onclick = () => download(blob, fileName);
    if (dlg.showModal) dlg.showModal(); else dlg.setAttribute("open", "");
  }
  document.addEventListener("DOMContentLoaded", () => {
    const dlg = $("#viewer");
    if (!dlg) return;
    $("#viewer-close").addEventListener("click", closeViewer);
    dlg.addEventListener("click", (e) => { if (e.target === dlg) closeViewer(); });
    dlg.addEventListener("close", () => { $("#viewer-body").innerHTML = ""; });
  });

  /** Decode base64 (no data: prefix) into a Blob. */
  function b64ToBlob(b64, type) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type });
  }

  /** Emit an app-wide event on document. */
  const emit = (name, detail) => document.dispatchEvent(new CustomEvent(name, { detail }));

  return { $, $$, esc, today, fmtDate, fmtBytes, fmtDuration, icon, toast, thumbnail, download,
    openViewer, closeViewer, b64ToBlob, emit };
})();

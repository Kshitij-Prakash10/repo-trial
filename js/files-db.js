/*
 * File storage for Health Diary: documents and voice notes in IndexedDB.
 *
 * The rest of the app touches IndexedDB only through this small API (like js/storage.js for
 * localStorage). Everything stays in this browser; nothing is uploaded.
 *
 * Database "healthDiary.files", three object stores:
 *   documents  { id, userId, title, category, docDate, tags: [], notes, link: "visit:<id>" | "condition:<id>" | "",
 *                fileName, mime, size (bytes), thumb (small JPEG data: URL or ""), source: "upload" | "scan",
 *                scan: null or { mode, method, modelVersion, confidence, pageFound } for scanned pages,
 *                createdAt, updatedAt (ISO 8601) }
 *   voiceNotes { id, userId, title, documentId, fileName, mime, size (bytes), duration (seconds or null),
 *                createdAt, updatedAt }
 *   blobs      { id, blob }   the file itself, same id as its documents / voiceNotes record
 *
 * Metadata and the file are separate so lists never load whole files into memory.
 */
const FilesDB = (() => {
  const DB_NAME = "healthDiary.files";
  const DB_VERSION = 1;
  const MAX_BYTES = 20 * 1024 * 1024; // per document or voice note

  const DOC_TYPES = { "application/pdf": "PDF", "image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WebP" };
  const AUDIO_TYPES = {
    "audio/webm": "WebM", "audio/ogg": "Ogg", "audio/mp4": "M4A", "audio/wav": "WAV", "audio/mpeg": "MP3",
  };

  let dbPromise = null;

  /** Open (and on first use create) the database. Rejects when IndexedDB is not available. */
  function open() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
      let req;
      try {
        if (!window.indexedDB) throw new Error("IndexedDB is not available in this browser");
        req = indexedDB.open(DB_NAME, DB_VERSION);
      } catch (err) {
        reject(err);
        return;
      }
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains("documents")) {
          db.createObjectStore("documents", { keyPath: "id" }).createIndex("userId", "userId");
        }
        if (!db.objectStoreNames.contains("voiceNotes")) {
          db.createObjectStore("voiceNotes", { keyPath: "id" }).createIndex("userId", "userId");
        }
        if (!db.objectStoreNames.contains("blobs")) db.createObjectStore("blobs", { keyPath: "id" });
      };
      req.onsuccess = () => {
        const db = req.result;
        db.onversionchange = () => { db.close(); dbPromise = null; };
        resolve(db);
      };
      req.onerror = () => reject(req.error || new Error("IndexedDB could not be opened"));
      req.onblocked = () => reject(new Error("IndexedDB is blocked by another tab of this app"));
    });
    dbPromise.catch(() => { dbPromise = null; });
    return dbPromise;
  }

  const reqP = (req) => new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  const txDone = (tx) => new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error("transaction failed"));
    tx.onabort = () => reject(tx.error || new Error("transaction aborted (storage may be full)"));
  });

  async function run(stores, mode, fn) {
    const db = await open();
    const tx = db.transaction(stores, mode);
    const result = fn(tx);
    await txDone(tx);
    return result;
  }

  async function listBy(store, userId) {
    const db = await open();
    const tx = db.transaction(store, "readonly");
    return reqP(tx.objectStore(store).index("userId").getAll(userId));
  }

  const now = () => new Date().toISOString();
  const uid = () => (typeof Storage !== "undefined" && Storage.uid ? Storage.uid()
    : Date.now().toString(36) + Math.random().toString(36).slice(2, 7));

  /* ---------- File type checks from the first bytes ---------- */

  async function head(blob, n = 16) {
    return new Uint8Array(await blob.slice(0, n).arrayBuffer());
  }
  const starts = (b, sig, at = 0) => sig.every((v, i) => b[at + i] === v);
  const ascii = (s) => [...s].map((c) => c.charCodeAt(0));

  /** MIME type of a document (PDF, PNG, JPEG, WebP) from its first bytes, or null. */
  async function sniffDocument(blob) {
    const b = await head(blob);
    if (starts(b, ascii("%PDF-"))) return "application/pdf";
    if (starts(b, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) return "image/png";
    if (starts(b, [0xff, 0xd8, 0xff])) return "image/jpeg";
    if (starts(b, ascii("RIFF")) && starts(b, ascii("WEBP"), 8)) return "image/webp";
    return null;
  }

  /** MIME type of an audio file (WebM, Ogg, MP4/M4A, WAV, MP3) from its first bytes, or null. */
  async function sniffAudio(blob) {
    const b = await head(blob);
    if (starts(b, [0x1a, 0x45, 0xdf, 0xa3])) return "audio/webm";
    if (starts(b, ascii("OggS"))) return "audio/ogg";
    if (starts(b, ascii("ftyp"), 4)) return "audio/mp4";
    if (starts(b, ascii("RIFF")) && starts(b, ascii("WAVE"), 8)) return "audio/wav";
    if (starts(b, ascii("ID3")) || (b[0] === 0xff && (b[1] & 0xe0) === 0xe0)) return "audio/mpeg";
    return null;
  }

  function checkSize(blob) {
    if (blob.size > MAX_BYTES) {
      const e = new Error(`The file is ${(blob.size / 1048576).toFixed(1)} MB. The limit is 20 MB.`);
      e.code = "too_large";
      throw e;
    }
    if (!blob.size) {
      const e = new Error("The file is empty.");
      e.code = "empty";
      throw e;
    }
  }

  /* ---------- Documents ---------- */

  /** Metadata of one user's documents (no file contents), newest first. */
  async function listDocuments(userId) {
    const docs = await listBy("documents", userId);
    return docs.sort((a, b) => (b.docDate || b.createdAt.slice(0, 10)).localeCompare(a.docDate || a.createdAt.slice(0, 10))
      || b.createdAt.localeCompare(a.createdAt));
  }

  /**
   * Store a document. The type is checked from the first bytes; size at most 20 MB.
   * meta: { title, category, docDate, tags, notes, link, fileName, thumb, source }. Returns the record.
   */
  async function addDocument(userId, meta, blob) {
    checkSize(blob);
    const mime = await sniffDocument(blob);
    if (!mime) {
      const e = new Error("This file is not a PDF, PNG, JPEG or WebP (checked from its contents).");
      e.code = "bad_type";
      throw e;
    }
    const t = now();
    const doc = {
      id: uid(), userId,
      title: (meta.title || "").trim() || (meta.fileName || "Document").replace(/\.[^.]+$/, ""),
      category: meta.category || "other",
      docDate: meta.docDate || "",
      tags: normTags(meta.tags),
      notes: meta.notes || "",
      link: meta.link || "",
      fileName: meta.fileName || "document",
      mime, size: blob.size,
      thumb: meta.thumb || "",
      source: meta.source || "upload",
      scan: meta.scan || null,
      createdAt: t, updatedAt: t,
    };
    const typed = blob.type === mime ? blob : new Blob([blob], { type: mime });
    await run(["documents", "blobs"], "readwrite", (tx) => {
      tx.objectStore("documents").put(doc);
      tx.objectStore("blobs").put({ id: doc.id, blob: typed });
    });
    return doc;
  }

  /** Change a document's metadata (title, category, docDate, tags, notes, link). Returns the record. */
  async function updateDocument(id, patch) {
    let out = null;
    await run(["documents"], "readwrite", (tx) => {
      const store = tx.objectStore("documents");
      const r = store.get(id);
      r.onsuccess = () => {
        if (!r.result) return;
        const allowed = ["title", "category", "docDate", "notes", "link"];
        out = { ...r.result };
        allowed.forEach((k) => { if (k in patch) out[k] = patch[k] ?? ""; });
        if ("tags" in patch) out.tags = normTags(patch.tags);
        if (!String(out.title).trim()) out.title = r.result.title;
        out.updatedAt = now();
        store.put(out);
      };
    });
    if (!out) throw new Error("That document no longer exists.");
    return out;
  }

  /** Delete a document and its file. Voice notes attached to it are kept and unlinked. */
  async function deleteDocument(id) {
    await run(["documents", "blobs", "voiceNotes"], "readwrite", (tx) => {
      tx.objectStore("documents").delete(id);
      tx.objectStore("blobs").delete(id);
      const vs = tx.objectStore("voiceNotes");
      const c = vs.openCursor();
      c.onsuccess = () => {
        const cur = c.result;
        if (!cur) return;
        if (cur.value.documentId === id) cur.update({ ...cur.value, documentId: "" });
        cur.continue();
      };
    });
  }

  /* ---------- Voice notes ---------- */

  /** Metadata of one user's voice notes, newest first. */
  async function listVoiceNotes(userId) {
    const notes = await listBy("voiceNotes", userId);
    return notes.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }

  /** Store a voice note. meta: { title, documentId, fileName, duration (s) }. Type checked from the bytes. */
  async function addVoiceNote(userId, meta, blob) {
    checkSize(blob);
    const mime = await sniffAudio(blob);
    if (!mime) {
      const e = new Error("This file is not a WebM, Ogg, M4A, WAV or MP3 audio file (checked from its contents).");
      e.code = "bad_type";
      throw e;
    }
    const t = now();
    const note = {
      id: uid(), userId,
      title: (meta.title || "").trim() || "Voice note",
      documentId: meta.documentId || "",
      fileName: meta.fileName || "voice-note",
      mime, size: blob.size,
      duration: Number.isFinite(meta.duration) ? Math.round(meta.duration * 10) / 10 : null,
      createdAt: t, updatedAt: t,
    };
    // Keep the recorder's own type (it carries the codec); fall back to the sniffed one.
    const typed = blob.type && blob.type.startsWith("audio/") ? blob : new Blob([blob], { type: mime });
    await run(["voiceNotes", "blobs"], "readwrite", (tx) => {
      tx.objectStore("voiceNotes").put(note);
      tx.objectStore("blobs").put({ id: note.id, blob: typed });
    });
    return note;
  }

  /** Rename a voice note or change its document link. patch: { title?, documentId? }. */
  async function updateVoiceNote(id, patch) {
    let out = null;
    await run(["voiceNotes"], "readwrite", (tx) => {
      const store = tx.objectStore("voiceNotes");
      const r = store.get(id);
      r.onsuccess = () => {
        if (!r.result) return;
        out = { ...r.result };
        if ("title" in patch && String(patch.title).trim()) out.title = String(patch.title).trim();
        if ("documentId" in patch) out.documentId = patch.documentId || "";
        out.updatedAt = now();
        store.put(out);
      };
    });
    if (!out) throw new Error("That voice note no longer exists.");
    return out;
  }

  /** Delete a voice note and its audio. */
  async function deleteVoiceNote(id) {
    await run(["voiceNotes", "blobs"], "readwrite", (tx) => {
      tx.objectStore("voiceNotes").delete(id);
      tx.objectStore("blobs").delete(id);
    });
  }

  /* ---------- Shared ---------- */

  /** The stored file (Blob) of a document or voice note, or null. */
  async function getBlob(id) {
    const db = await open();
    const r = await reqP(db.transaction("blobs", "readonly").objectStore("blobs").get(id));
    return r ? r.blob : null;
  }

  /** Delete every document, voice note and file of one user. */
  async function deleteUserFiles(userId) {
    await run(["documents", "voiceNotes", "blobs"], "readwrite", (tx) => {
      ["documents", "voiceNotes"].forEach((name) => {
        const store = tx.objectStore(name);
        const c = store.index("userId").openCursor(IDBKeyRange.only(userId));
        c.onsuccess = () => {
          const cur = c.result;
          if (!cur) return;
          tx.objectStore("blobs").delete(cur.value.id);
          cur.delete();
          cur.continue();
        };
      });
    });
  }

  /**
   * Storage used by one user, in bytes: { documents: {count, bytes}, voiceNotes: {count, bytes}, bytes }.
   * Counts file sizes plus thumbnails.
   */
  async function usage(userId) {
    const [docs, notes] = await Promise.all([listBy("documents", userId), listBy("voiceNotes", userId)]);
    const d = docs.reduce((s, x) => s + x.size + (x.thumb ? x.thumb.length : 0), 0);
    const v = notes.reduce((s, x) => s + x.size, 0);
    return { documents: { count: docs.length, bytes: d }, voiceNotes: { count: notes.length, bytes: v }, bytes: d + v };
  }

  /** The browser's own estimate for this site, { usage, quota } in bytes, or null when not supported. */
  async function browserEstimate() {
    try {
      if (navigator.storage && navigator.storage.estimate) return await navigator.storage.estimate();
    } catch (err) { /* not supported */ }
    return null;
  }

  function normTags(v) {
    const list = Array.isArray(v) ? v : String(v || "").split(",");
    const out = [];
    list.map((t) => String(t).trim()).filter(Boolean).forEach((t) => {
      if (!out.some((o) => o.toLowerCase() === t.toLowerCase())) out.push(t.slice(0, 40));
    });
    return out.slice(0, 20);
  }

  return {
    MAX_BYTES, DOC_TYPES, AUDIO_TYPES,
    open, sniffDocument, sniffAudio,
    listDocuments, addDocument, updateDocument, deleteDocument,
    listVoiceNotes, addVoiceNote, updateVoiceNote, deleteVoiceNote,
    getBlob, deleteUserFiles, usage, browserEstimate,
  };
})();

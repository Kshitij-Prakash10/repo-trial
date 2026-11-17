/*
 * Storage layer for Health Diary.
 *
 * Every user is stored separately in localStorage. Swap this file for an
 * API-backed implementation later without touching the UI code.
 *
 * Data shape for one user:
 * {
 *   id, createdAt,
 *   profile:     { name, dob, gender, bloodGroup, emergency },
 *   vitals:      [{ id, date, height, weight, bp, heartRate, sugar, notes }],
 *   conditions:  [{ id, name, diagnosed, status, notes }],
 *   medications: [{ id, name, dosage, start, end, reason }],
 *   allergies:   [{ id, name, severity, reaction }],
 *   visits:      [{ id, date, type, doctor, notes }]
 * }
 */
const Storage = (() => {
  const KEY = "healthDiary.users";
  const SESSION_KEY = "healthDiary.currentUser";

  const BACKUP_KEY = "healthDiary.users.unreadable"; // a copy of saved data that could not be read

  const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

  /** The saved users as { ok, users }. ok is false when something is saved but cannot be read. */
  function load() {
    const raw = localStorage.getItem(KEY);
    if (raw === null) return { ok: true, users: {} };
    try {
      const users = JSON.parse(raw);
      if (isObject(users)) return { ok: true, users };
    } catch { /* fall through */ }
    return { ok: false, users: {} };
  }

  const read = () => {
    const users = load().users;
    // Tolerate entries from older versions or edited by hand: skip non-objects, give every user a name.
    Object.keys(users).forEach((id) => {
      const u = users[id];
      if (!isObject(u)) { delete users[id]; return; }
      if (!isObject(u.profile)) u.profile = {};
      u.profile.name = String(u.profile.name ?? "").trim() || "Unnamed";
      if (u.id == null) u.id = id;
    });
    return users;
  };

  /**
   * Save all users. Refuses (throws) while the saved data cannot be read, so that one damaged value is
   * never replaced by an almost empty one. Also throws when the browser's storage is full.
   */
  const write = (users) => {
    if (!load().ok) throw new Error("The saved diary data cannot be read, so nothing was changed.");
    localStorage.setItem(KEY, JSON.stringify(users));
  };
  const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

  return {
    uid,

    /** True when saved data exists but cannot be read (damaged). Nothing is written while this is true. */
    damaged() { return !load().ok; },

    /** The unreadable saved data exactly as stored (for the user to download), or null. */
    damagedData() { return load().ok ? null : localStorage.getItem(KEY); },

    /** Set the unreadable data aside (kept under a backup key when it fits) and start with no users. */
    setDamagedAside() {
      if (load().ok) return;
      try { localStorage.setItem(BACKUP_KEY, localStorage.getItem(KEY)); } catch { /* no room for a copy */ }
      localStorage.removeItem(KEY);
    },

    listUsers() {
      return Object.values(read()).sort((a, b) =>
        a.profile.name.localeCompare(b.profile.name));
    },

    getUser(id) { return read()[id] || null; },

    createUser(profile) {
      const users = read();
      const user = {
        id: uid(),
        createdAt: new Date().toISOString(),
        profile,
        vitals: [], conditions: [], medications: [], allergies: [], visits: [],
      };
      users[user.id] = user;
      write(users);
      return user;
    },

    saveUser(user) {
      const users = read();
      users[user.id] = user;
      write(users);
    },

    deleteUser(id) {
      const users = read();
      delete users[id];
      write(users);
    },

    getCurrentUserId() { return localStorage.getItem(SESSION_KEY); },
    setCurrentUserId(id) {
      if (id) localStorage.setItem(SESSION_KEY, id);
      else localStorage.removeItem(SESSION_KEY);
    },
  };
})();

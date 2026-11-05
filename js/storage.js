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

  const read = () => {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; }
    catch { return {}; }
  };
  const write = (users) => localStorage.setItem(KEY, JSON.stringify(users));
  const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

  return {
    uid,

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

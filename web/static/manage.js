/* Nova — Manage People page */
(() => {
  "use strict";

  const manageEmpty  = document.getElementById("manageEmpty");
  const manageList   = document.getElementById("manageList");

  const addPersonBtn   = document.getElementById("addPersonBtn");
  const addPersonForm  = document.getElementById("addPersonForm");
  const addPersonError = document.getElementById("addPersonError");
  const saveNewPerson  = document.getElementById("saveNewPerson");
  const cancelNewPerson= document.getElementById("cancelNewPerson");

  let managePeople = [];
  const voiceCounts = {};   // person_id -> sample count for the open voice enroll panel

  async function api(path, opts) {
    const res = await fetch(path, opts);
    return res.json();
  }
  const jsonHeaders = { "Content-Type": "application/json" };

  async function loadManagePeople() {
    const data = await api("/api/people").catch(() => null);
    if (!data || !data.ok) return;
    managePeople = data.people;
    renderManageList();
  }

  function personRowHtml(p) {
    return `
      <div class="manage-card" data-id="${p.id}">
        <div class="manage-card-top">
          <div>
            <div class="manage-name">${p.first_name} ${p.last_name}</div>
            <div class="manage-meta">
              <span>${p.role} \u00b7 ${p.access_level} access</span>
              <span class="dot-tag ${p.has_face ? "on" : ""}"><span class="dot"></span>face</span>
              <span class="dot-tag ${p.has_voice ? "on" : ""}"><span class="dot"></span>voice</span>
              <span class="dot-tag ${p.has_pin ? "on" : ""}"><span class="dot"></span>pin</span>
            </div>
          </div>
          <button class="icon-btn delete-person-btn" data-id="${p.id}" title="Remove">\u2715</button>
        </div>

        <div class="train-section">
          <div class="train-hint">Train from files already on disk — a
            folder with photos/videos and/or audio in it.</div>
          <div class="train-row">
            <input type="text" class="train-path-input" data-id="${p.id}"
                   placeholder="e.g. known_faces/${p.first_name.toLowerCase()}  (photos, videos, and/or audio together)">
          </div>
          <div class="train-row">
            <button class="chip-btn split-path-btn" data-id="${p.id}">Use separate paths for face &amp; voice</button>
          </div>
          <div class="train-row split-paths" id="split-paths-${p.id}" hidden>
            <input type="text" class="train-face-path-input" data-id="${p.id}" placeholder="Face folder (photos/videos)">
            <input type="text" class="train-voice-path-input" data-id="${p.id}" placeholder="Voice folder (audio files)">
          </div>
          <div class="train-row">
            <label class="checkbox-label">
              <input type="checkbox" class="append-checkbox" data-id="${p.id}">
              Add to existing data instead of replacing it
            </label>
          </div>
          <div class="train-row">
            <button class="btn-small btn-train train-btn" data-id="${p.id}">Train from Path</button>
            <span class="enroll-status" id="train-status-${p.id}"></span>
          </div>
        </div>

        <div class="manage-card-actions">
          <button class="chip-btn toggle-face-btn" data-id="${p.id}">Or Record Face Live</button>
          <button class="chip-btn toggle-voice-btn" data-id="${p.id}">Or Record Voice Live</button>
          <button class="chip-btn toggle-pin-btn" data-id="${p.id}">Set PIN</button>
        </div>
        <div class="enroll-panel" id="pin-enroll-${p.id}" hidden>
          <input type="password" inputmode="numeric" maxlength="8"
                 class="pin-set-input" data-id="${p.id}"
                 placeholder="4-8 digit PIN">
          <button class="btn-small set-pin-btn" data-id="${p.id}">Save PIN</button>
          ${p.has_pin ? `<button class="btn-small btn-ghost remove-pin-btn" data-id="${p.id}">Remove PIN</button>` : ""}
          <span class="enroll-status" id="pin-set-status-${p.id}"></span>
        </div>
        <div class="enroll-panel" id="face-enroll-${p.id}" hidden>
          <button class="btn-small capture-face-btn" data-id="${p.id}">Capture from Camera</button>
          <label class="upload-label">Upload Photo
            <input type="file" accept="image/*" class="upload-face-input" data-id="${p.id}">
          </label>
          <span class="enroll-status" id="face-status-${p.id}"></span>
        </div>
        <div class="enroll-panel" id="voice-enroll-${p.id}" hidden>
          <button class="btn-small record-sample-btn" data-id="${p.id}">
            Record Sample (<span id="voice-count-${p.id}">${voiceCounts[p.id] || 0}</span>)
          </button>
          <button class="btn-small finish-voice-btn" data-id="${p.id}">Finish</button>
          <button class="btn-small btn-ghost cancel-voice-btn" data-id="${p.id}">Cancel</button>
          <span class="enroll-status" id="voice-status-${p.id}"></span>
        </div>
      </div>`;
  }

  function renderManageList() {
    manageEmpty.hidden = managePeople.length > 0;
    manageList.innerHTML = managePeople.map(personRowHtml).join("");
  }

  function setEnrollStatus(el, text, kind) {
    el.textContent = text;
    el.classList.remove("ok", "err");
    if (kind) el.classList.add(kind);
  }

  // ── Add person form ──────────────────────────────────────────────────
  addPersonBtn.addEventListener("click", () => {
    addPersonForm.hidden = !addPersonForm.hidden;
    addPersonError.hidden = true;
  });
  cancelNewPerson.addEventListener("click", () => {
    addPersonForm.hidden = true;
  });
  saveNewPerson.addEventListener("click", async () => {
    const first = document.getElementById("newFirstName").value.trim();
    const last  = document.getElementById("newLastName").value.trim();
    const role  = document.getElementById("newRole").value;
    const access= document.getElementById("newAccess").value;
    const notes = document.getElementById("newNotes").value.trim();

    if (!first || !last) {
      addPersonError.textContent = "First and last name are required.";
      addPersonError.hidden = false;
      return;
    }
    saveNewPerson.disabled = true;
    const res = await api("/api/people", {
      method: "POST", headers: jsonHeaders,
      body: JSON.stringify({ first_name: first, last_name: last,
                              role, access_level: access, notes }),
    }).catch(() => ({ ok: false, error: "Network error" }));
    saveNewPerson.disabled = false;

    if (!res.ok) {
      addPersonError.textContent = res.error || "Could not add person.";
      addPersonError.hidden = false;
      return;
    }
    document.getElementById("newFirstName").value = "";
    document.getElementById("newLastName").value = "";
    document.getElementById("newNotes").value = "";
    addPersonForm.hidden = true;
    loadManagePeople();
  });

  // ── Row actions (event delegation) ───────────────────────────────────
  manageList.addEventListener("click", async (e) => {
    const del = e.target.closest(".delete-person-btn");
    if (del) {
      const id = del.dataset.id;
      const person = managePeople.find(p => String(p.id) === id);
      const label = person ? `${person.first_name} ${person.last_name}` : "this person";
      if (!confirm(`Remove ${label}? This deletes their face, voice, and clothing data.`)) return;
      await api(`/api/people/${id}`, { method: "DELETE" });
      loadManagePeople();
      return;
    }

    const splitToggle = e.target.closest(".split-path-btn");
    if (splitToggle) {
      const panel = document.getElementById(`split-paths-${splitToggle.dataset.id}`);
      panel.hidden = !panel.hidden;
      splitToggle.classList.toggle("active", !panel.hidden);
      splitToggle.textContent = panel.hidden
        ? "Use separate paths for face & voice"
        : "Use one combined path instead";
      return;
    }

    const train = e.target.closest(".train-btn");
    if (train) {
      const id = train.dataset.id;
      const statusEl = document.getElementById(`train-status-${id}`);
      const splitPanel = document.getElementById(`split-paths-${id}`);
      const append = document.querySelector(`.append-checkbox[data-id="${id}"]`).checked;

      let facePath, voicePath;
      if (splitPanel.hidden) {
        const combined = document.querySelector(`.train-path-input[data-id="${id}"]`).value.trim();
        facePath = combined;
        voicePath = combined;
      } else {
        facePath  = document.querySelector(`.train-face-path-input[data-id="${id}"]`).value.trim();
        voicePath = document.querySelector(`.train-voice-path-input[data-id="${id}"]`).value.trim();
      }

      if (!facePath && !voicePath) {
        setEnrollStatus(statusEl, "Enter a folder path first.", "err");
        return;
      }

      train.disabled = true;
      setEnrollStatus(statusEl, "Training\u2026");
      const results = [];

      if (facePath) {
        const r = await api(`/api/people/${id}/face/enroll-path`, {
          method: "POST", headers: jsonHeaders,
          body: JSON.stringify({ path: facePath, append }),
        }).catch(() => ({ ok: false, error: "Network error" }));
        results.push(r.ok ? `Face: ${r.count} sample(s)` : `Face: ${r.error}`);
      }
      if (voicePath) {
        const r = await api(`/api/people/${id}/voice/enroll-path`, {
          method: "POST", headers: jsonHeaders,
          body: JSON.stringify({ path: voicePath, append }),
        }).catch(() => ({ ok: false, error: "Network error" }));
        results.push(r.ok ? `Voice: ${r.count} sample(s)` : `Voice: ${r.error}`);
      }

      train.disabled = false;
      // a failed line's message won't end in "sample(s)"
      const failed = results.some(r => !r.endsWith("sample(s)"));
      setEnrollStatus(statusEl, results.join("  \u00b7  "), failed ? "err" : "ok");
      loadManagePeople();
      return;
    }

    const toggleFace = e.target.closest(".toggle-face-btn");
    if (toggleFace) {
      const panel = document.getElementById(`face-enroll-${toggleFace.dataset.id}`);
      panel.hidden = !panel.hidden;
      return;
    }
    const toggleVoice = e.target.closest(".toggle-voice-btn");
    if (toggleVoice) {
      const panel = document.getElementById(`voice-enroll-${toggleVoice.dataset.id}`);
      panel.hidden = !panel.hidden;
      return;
    }

    const togglePin = e.target.closest(".toggle-pin-btn");
    if (togglePin) {
      const panel = document.getElementById(`pin-enroll-${togglePin.dataset.id}`);
      panel.hidden = !panel.hidden;
      return;
    }

    const setPin = e.target.closest(".set-pin-btn");
    if (setPin) {
      const id = setPin.dataset.id;
      const input = document.querySelector(`.pin-set-input[data-id="${id}"]`);
      const statusEl = document.getElementById(`pin-set-status-${id}`);
      const pin = input.value.trim();
      setPin.disabled = true;
      const res = await api(`/api/people/${id}/pin`, {
        method: "POST", headers: jsonHeaders, body: JSON.stringify({ pin }),
      }).catch(() => ({ ok: false, error: "Network error" }));
      setPin.disabled = false;
      if (res.ok) {
        setEnrollStatus(statusEl, "PIN saved.", "ok");
        input.value = "";
        loadManagePeople();
      } else {
        setEnrollStatus(statusEl, res.error || "Could not save PIN.", "err");
      }
      return;
    }

    const removePin = e.target.closest(".remove-pin-btn");
    if (removePin) {
      const id = removePin.dataset.id;
      if (!confirm("Remove this person's PIN?")) return;
      await api(`/api/people/${id}/pin`, { method: "DELETE" });
      loadManagePeople();
      return;
    }

    const capture = e.target.closest(".capture-face-btn");
    if (capture) {
      const id = capture.dataset.id;
      const statusEl = document.getElementById(`face-status-${id}`);
      capture.disabled = true;
      setEnrollStatus(statusEl, "Capturing\u2026");
      const res = await api(`/api/people/${id}/face/capture`, { method: "POST" })
        .catch(() => ({ ok: false, error: "Network error" }));
      capture.disabled = false;
      if (res.ok) {
        setEnrollStatus(statusEl, `Saved \u2014 ${res.count} sample(s) total.`, "ok");
        loadManagePeople();
      } else {
        setEnrollStatus(statusEl, res.error || "Capture failed.", "err");
      }
      return;
    }

    const record = e.target.closest(".record-sample-btn");
    if (record) {
      const id = record.dataset.id;
      const statusEl = document.getElementById(`voice-status-${id}`);
      const countEl  = document.getElementById(`voice-count-${id}`);
      record.disabled = true;
      setEnrollStatus(statusEl, "Recording \u2014 speak now\u2026");
      const res = await api(`/api/people/${id}/voice/sample`, { method: "POST" })
        .catch(() => ({ ok: false, error: "Network error" }));
      record.disabled = false;
      if (res.ok) {
        voiceCounts[id] = res.count;
        countEl.textContent = res.count;
        setEnrollStatus(statusEl, "Sample captured.", "ok");
      } else {
        if (typeof res.count === "number") {
          voiceCounts[id] = res.count;
          countEl.textContent = res.count;
        }
        setEnrollStatus(statusEl, res.error || "Recording failed.", "err");
      }
      return;
    }

    const finish = e.target.closest(".finish-voice-btn");
    if (finish) {
      const id = finish.dataset.id;
      const statusEl = document.getElementById(`voice-status-${id}`);
      finish.disabled = true;
      const res = await api(`/api/people/${id}/voice/finish`, { method: "POST" })
        .catch(() => ({ ok: false, error: "Network error" }));
      finish.disabled = false;
      if (res.ok) {
        setEnrollStatus(statusEl, `Enrolled \u2014 ${res.count} sample(s) saved.`, "ok");
        delete voiceCounts[id];
        loadManagePeople();
      } else {
        setEnrollStatus(statusEl, res.error || "Could not save.", "err");
      }
      return;
    }

    const cancel = e.target.closest(".cancel-voice-btn");
    if (cancel) {
      const id = cancel.dataset.id;
      await api(`/api/people/${id}/voice/cancel`, { method: "POST" });
      delete voiceCounts[id];
      document.getElementById(`voice-count-${id}`).textContent = "0";
      setEnrollStatus(document.getElementById(`voice-status-${id}`), "Cancelled.");
      return;
    }
  });

  manageList.addEventListener("change", async (e) => {
    const upload = e.target.closest(".upload-face-input");
    if (!upload || !upload.files.length) return;
    const id = upload.dataset.id;
    const statusEl = document.getElementById(`face-status-${id}`);
    setEnrollStatus(statusEl, "Uploading\u2026");
    const form = new FormData();
    form.append("file", upload.files[0]);
    const res = await fetch(`/api/people/${id}/face/upload`, { method: "POST", body: form })
      .then(r => r.json())
      .catch(() => ({ ok: false, error: "Network error" }));
    upload.value = "";
    if (res.ok) {
      setEnrollStatus(statusEl, `Saved \u2014 ${res.count} sample(s) total.`, "ok");
      loadManagePeople();
    } else {
      setEnrollStatus(statusEl, res.error || "Upload failed.", "err");
    }
  });

  loadManagePeople();
})();

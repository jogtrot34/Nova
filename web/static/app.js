/* Nova web UI — WebSocket client + waveform + panels */
(() => {
  "use strict";

  const scene       = document.getElementById("scene");
  const connPill     = document.getElementById("connPill");
  const connDot      = document.getElementById("connDot");
  const connText     = document.getElementById("connText");
  const voicePill    = document.getElementById("voicePill");
  const voiceDot     = document.getElementById("voiceDot");
  const voiceText    = document.getElementById("voiceText");
  const stateLabel   = document.getElementById("stateLabel");
  const statusText   = document.getElementById("statusText");
  const promptText   = document.getElementById("promptText");
  const waveStage    = document.getElementById("waveStage");
  const canvas       = document.getElementById("waveCanvas");
  const ctx          = canvas.getContext("2d");

  const alarmFlash      = document.getElementById("alarmFlash");
  const silenceAlarmBtn = document.getElementById("silenceAlarmBtn");

  const peopleList   = document.getElementById("peopleList");
  const peopleEmpty  = document.getElementById("peopleEmpty");
  const peopleBadge  = document.getElementById("peopleBadge");

  const eventsList   = document.getElementById("eventsList");
  const eventsEmpty  = document.getElementById("eventsEmpty");

  const statusList   = document.getElementById("statusList");

  const confEmpty    = document.getElementById("confEmpty");
  const ringsRow      = document.getElementById("ringsRow");
  const ringFace       = document.getElementById("ringFace");
  const ringVoice       = document.getElementById("ringVoice");
  const ringCombined     = document.getElementById("ringCombined");

  const cameraImg     = document.getElementById("cameraImg");
  const cameraFallback = document.getElementById("cameraFallback");
  const liveDot        = document.getElementById("liveDot");

  // ── State labels shown above the waveform ──────────────────────────────
  const STATE_LABEL = {
    idle:        "MONITORING",
    detected:    "PERSON DETECTED",
    identifying: "IDENTIFYING\u2026",
    speaking:    "NOVA SPEAKING",
    confirmed:   "IDENTITY CONFIRMED",
    waiting:     "VOICE VERIFICATION",
    denied:      "UNKNOWN / DENIED",
    wake:        "WAKE WORD HEARD",
    pin_required:"PIN REQUIRED",
    safe_word:   "SAFE WORD ARMED",
    alarm:       "\u26A0 INTRUDER ALARM \u26A0",
  };

  const STATE_PROMPT = {
    idle:        "Say \u201cNova\u201d to wake me",
    wake:        "How can I help you?",
    speaking:    "Nova is speaking\u2026",
    confirmed:   "Access granted",
    waiting:     "Say something to verify\u2026",
    denied:      "Access denied",
    detected:    "Taking a closer look\u2026",
    identifying: "Matching face &amp; voice\u2026",
    pin_required:"Not quite sure \u2014 enter your PIN below",
    safe_word:   "Say it again within 30s to send an alert",
    alarm:       "Unidentified person alone in view \u2014 alerting emergency contacts",
  };

  // ── Local UI state ──────────────────────────────────────────────────────
  const people = new Map();   // track_id -> payload
  let orbState = "idle";
  let amplitude = 0.0;
  let targetAmplitude = 0.0;

  // ═══════════════════════════════════════════════════════════════════════
  // Waveform (canvas equalizer)
  // ═══════════════════════════════════════════════════════════════════════
  const N_BARS = 46;
  let bars = new Array(N_BARS).fill(0.06);
  let dpr = Math.max(1, window.devicePixelRatio || 1);

  function resizeCanvas() {
    dpr = Math.max(1, window.devicePixelRatio || 1);
    const rect = waveStage.getBoundingClientRect();
    canvas.width  = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    canvas.style.width  = rect.width + "px";
    canvas.style.height = rect.height + "px";
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  function accentColor() {
    return getComputedStyle(scene).getPropertyValue("--accent").trim() || "#22d3ee";
  }
  let cachedColor = accentColor();

  let tick = 0;
  function drawWave() {
    tick += 1;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // amplitude smoothing (attack fast, release slow)
    amplitude += (targetAmplitude - amplitude) * (targetAmplitude > amplitude ? 0.35 : 0.06);

    // ── Vibration — the whole wave stage shakes in proportion to how
    // loud whatever the mic just picked up was, so it *feels* like it's
    // reacting to sound rather than just showing a state. Small deadzone
    // so normal room-tone doesn't cause a constant idle jitter.
    const VIBRATE_DEADZONE = 0.16;
    const VIBRATE_MAX_PX   = 5;
    const excess = Math.max(0, amplitude - VIBRATE_DEADZONE);
    const jitterMag = Math.min(excess * 9, 1) * VIBRATE_MAX_PX;
    if (jitterMag > 0.1) {
      const jx = (Math.random() - 0.5) * 2 * jitterMag;
      const jy = (Math.random() - 0.5) * 2 * jitterMag;
      waveStage.style.transform = `translate(${jx.toFixed(2)}px, ${jy.toFixed(2)}px)`;
    } else if (waveStage.style.transform) {
      waveStage.style.transform = "";
    }

    const speaking = (orbState === "speaking" || orbState === "wake");
    const identifying = (orbState === "identifying");
    const baseSpeed = speaking ? 0.14 : identifying ? 0.10 : 0.045;

    const midY = h / 2;
    const gap = w / N_BARS;
    const barW = gap * 0.52 * dpr;

    for (let i = 0; i < N_BARS; i++) {
      const phase = (i / N_BARS) * Math.PI * 2;
      const idleWave = (Math.sin(tick * baseSpeed + phase * 2.2) * 0.5 + 0.5);
      const ampWave  = amplitude * (0.55 + 0.45 * Math.sin(tick * 0.22 + phase * 3.1));
      let level = 0.10 + idleWave * 0.18 + ampWave * 0.9;
      level = Math.max(0.06, Math.min(1, level));
      bars[i] = bars[i] * 0.55 + level * 0.45;

      const barH = bars[i] * (h * 0.42);
      const x = i * gap * dpr + (gap * dpr - barW) / 2;

      const grad = ctx.createLinearGradient(0, midY - barH, 0, midY + barH);
      grad.addColorStop(0, cachedColor + "cc");
      grad.addColorStop(0.5, cachedColor + "ff");
      grad.addColorStop(1, cachedColor + "cc");
      ctx.fillStyle = grad;

      roundRect(ctx, x, midY - barH, barW, barH * 2, barW / 2);
      ctx.fill();
    }

    requestAnimationFrame(drawWave);
  }

  function roundRect(c, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y,     x + w, y + h, r);
    c.arcTo(x + w, y + h, x,     y + h, r);
    c.arcTo(x,     y + h, x,     y,     r);
    c.arcTo(x,     y,     x + w, y,     r);
    c.closePath();
  }

  requestAnimationFrame(drawWave);

  // ═══════════════════════════════════════════════════════════════════════
  // Orb / hero state
  // ═══════════════════════════════════════════════════════════════════════
  function setOrbState(state) {
    orbState = state;
    scene.dataset.orb = state;
    stateLabel.textContent = STATE_LABEL[state] || state.toUpperCase();
    cachedColor = accentColor();

    if (state === "wake" || state === "speaking") {
      targetAmplitude = Math.max(targetAmplitude, 0.35);
    }

    alarmFlash.classList.toggle("on", state === "alarm");
    silenceAlarmBtn.hidden = state !== "alarm";

    if (!statusOverridden) {
      promptText.innerHTML = STATE_PROMPT[state] || "Monitoring\u2026";
    }
  }

  let statusOverridden = false;
  function setStatusText(text) {
    statusText.textContent = text;
    promptText.textContent = text;
    statusOverridden = true;
    clearTimeout(setStatusText._t);
    setStatusText._t = setTimeout(() => { statusOverridden = false; }, 4000);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // System status panel (static rows, driven by connection + orb state)
  // ═══════════════════════════════════════════════════════════════════════
  function renderStatusList() {
    const rows = [
      { label: "System Active",       on: true },
      { label: connText.textContent === "Live" ? "Listening\u2026" : "Reconnecting\u2026", on: connText.textContent === "Live" },
      { label: "Camera Online",       on: !cameraFallback.hasAttribute("hidden") === false },
      { label: "Microphone Online",   on: true },
    ];
    statusList.innerHTML = rows.map(r => `
      <li>
        <span class="s-left">
          <span class="status-dot ${r.on ? "on" : "off"}"></span>
          ${r.label}
        </span>
        <span class="status-time">${new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}</span>
      </li>`).join("");
  }
  setInterval(renderStatusList, 15000);

  // ═══════════════════════════════════════════════════════════════════════
  // Detected people panel
  // ═══════════════════════════════════════════════════════════════════════
  function personClass(p) {
    if (p.status === "lost") return "lost";
    if (p.conflict) return "denied";
    if (p.decision === "granted") return "granted";
    if (p.pin_required) return "pin";
    if (p.needs_voice) return "waiting";
    return "denied";
  }
  function personBadgeText(p) {
    if (p.status === "lost") return "Last Seen";
    if (p.conflict) return "Conflict";
    if (p.decision === "granted") return "Seen Now";
    if (p.pin_required) return "Enter PIN";
    if (p.needs_voice) return "Tracking\u2026";
    return "Tracking\u2026";
  }
  function methodText(p) {
    if (p.method === "face + voice") return "Face + Voice Match";
    if (p.method === "face") return "Face Only";
    if (p.method === "voice") return "Voice Only";
    if (p.method === "face + voice + clothing") return "Face + Voice + Clothing";
    if (p.method && p.method.includes("pin")) return "Confirmed by PIN";
    return p.name === "Unknown" ? "Unidentified" : (p.method || "Analyzing\u2026");
  }
  function timeAgo(ts) {
    const secs = Math.max(0, Math.round(Date.now() / 1000 - ts));
    return secs < 60 ? `${secs} sec ago` : `${Math.round(secs / 60)} min ago`;
  }

  function renderPeople() {
    const list = Array.from(people.values())
      .sort((a, b) => (b.combined_conf || 0) - (a.combined_conf || 0));

    peopleBadge.textContent = list.filter(p => p.status !== "lost").length;
    peopleEmpty.hidden = list.length > 0;

    peopleList.innerHTML = list.map(p => {
      const cls = personClass(p);
      const icon = `<svg viewBox="0 0 24 24"><path d="M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10Zm0 2c-4.4 0-8 2.24-8 5v2h16v-2c0-2.76-3.6-5-8-5Z"/></svg>`;
      const displayName = p.name !== "Unknown" ? p.name
        : (p.candidate_name ? `${p.candidate_name}?` : "Unknown");
      const meta = cls === "lost"
        ? `Lost Sight`
        : (p.name === "Unknown" ? (p.candidate_name ? "Unconfirmed match" : "Face Only")
                                 : methodText(p));
      const sub = cls === "lost"
        ? timeAgo(p.last_seen)
        : `${p.top_desc || ""}${p.bottom_desc ? " \u00b7 " + p.bottom_desc : ""}`;
      const ambiguousNote = (p.ambiguous && p.runner_up_name)
        ? `<div class="person-ambiguous">Close call \u2014 also resembles ${p.runner_up_name}</div>`
        : "";
      const pinBox = p.pin_required ? `
          <div class="pin-box">
            <input type="password" inputmode="numeric" maxlength="8"
                   class="pin-input" data-track="${p.track_id}"
                   data-person="${p.candidate_person_id || ""}"
                   placeholder="PIN">
            <button class="btn-small pin-submit" data-track="${p.track_id}"
                    data-person="${p.candidate_person_id || ""}">Confirm</button>
            <span class="pin-status" id="pin-status-${p.track_id}"></span>
          </div>` : "";
      return `
        <div class="person-card ${cls}" data-track="${p.track_id}">
          <div class="avatar">${icon}</div>
          <div class="person-info">
            <div class="person-name">${displayName}</div>
            <div class="person-meta ${cls}">${meta}</div>
            <div class="person-sub">${sub}</div>
            ${ambiguousNote}
            ${pinBox}
          </div>
          <div class="person-right">
            <div class="person-conf">${Math.round((p.combined_conf || 0) * 100)}%</div>
            <div class="person-badge ${cls}">${personBadgeText(p)}</div>
          </div>
        </div>`;
    }).join("");

    renderConfidenceRings(list);
  }

  // ── PIN entry (inline on a Detected People card) ─────────────────────
  peopleList.addEventListener("click", async (e) => {
    const btn = e.target.closest(".pin-submit");
    if (!btn) return;
    const trackId  = btn.dataset.track;
    const personId = btn.dataset.person;
    const input    = peopleList.querySelector(`.pin-input[data-track="${trackId}"]`);
    const statusEl = document.getElementById(`pin-status-${trackId}`);
    const pin = input ? input.value.trim() : "";

    if (!personId) {
      statusEl.textContent = "No candidate to check against.";
      statusEl.className = "pin-status err";
      return;
    }
    if (!pin) {
      statusEl.textContent = "Enter a PIN first.";
      statusEl.className = "pin-status err";
      return;
    }

    btn.disabled = true;
    statusEl.textContent = "Checking\u2026";
    statusEl.className = "pin-status";

    const res = await fetch("/api/verify-pin", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track_id: Number(trackId),
                              person_id: Number(personId), pin }),
    }).then(r => r.json()).catch(() => ({ ok: false, error: "Network error" }));

    btn.disabled = false;

    if (res.ok && res.granted) {
      statusEl.textContent = "Confirmed.";
      statusEl.className = "pin-status ok";
    } else {
      statusEl.textContent = res.error || "Incorrect PIN.";
      statusEl.className = "pin-status err";
      if (input) input.value = "";
    }
  });

  peopleList.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.classList.contains("pin-input")) {
      const trackId = e.target.dataset.track;
      const btn = peopleList.querySelector(`.pin-submit[data-track="${trackId}"]`);
      if (btn) btn.click();
    }
  });

  function renderConfidenceRings(list) {
    const primary = list.find(p => p.status !== "lost" && p.name !== "Unknown") || list[0];
    if (!primary) {
      confEmpty.hidden = false;
      ringsRow.hidden = true;
      return;
    }
    confEmpty.hidden = true;
    ringsRow.hidden = false;
    setRing(ringFace, primary.face_conf);
    setRing(ringVoice, primary.voice_conf);
    setRing(ringCombined, primary.combined_conf);
  }
  function setRing(el, value) {
    const pct = Math.round((value || 0) * 100);
    el.style.setProperty("--pct", pct);
    el.querySelector(".ring-val").textContent = pct + "%";
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Events panel
  // ═══════════════════════════════════════════════════════════════════════
  const EVENT_ICON = {
    face:     `<path d="M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10Zm0 2c-4.4 0-8 2.24-8 5v2h16v-2c0-2.76-3.6-5-8-5Z"/>`,
    voice:    `<path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.93V21h2v-2.07A7 7 0 0 0 19 12h-2Z"/>`,
    identity: `<path d="m9 16.2-3.5-3.5-1.4 1.4L9 19 20 8l-1.4-1.4L9 16.2Z"/>`,
    wake:     `<path d="M3 12h3l2-7 4 14 3-9 2 4h4"/>`,
    info:     `<path d="M11 7h2v2h-2Zm0 4h2v6h-2ZM12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Z"/>`,
  };

  function addEventRow(ev, prepend) {
    eventsEmpty.hidden = true;
    const row = document.createElement("div");
    row.className = "event-row";
    row.innerHTML = `
      <div class="event-ic ${ev.kind}"><svg viewBox="0 0 24 24">${EVENT_ICON[ev.kind] || EVENT_ICON.info}</svg></div>
      <div>
        <div class="event-title">${ev.title}</div>
        ${ev.subtitle ? `<div class="event-sub">${ev.subtitle}</div>` : ""}
      </div>
      <div class="event-time" style="margin-left:auto">${ev.ts}</div>
    `;
    row.querySelector("div:nth-child(2)").style.flex = "1";
    if (prepend) eventsList.prepend(row); else eventsList.appendChild(row);
    while (eventsList.children.length > 40) eventsList.removeChild(eventsList.lastChild);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Panel collapse / expand
  // ═══════════════════════════════════════════════════════════════════════
  document.querySelectorAll(".panel-head").forEach(btn => {
    btn.addEventListener("click", () => {
      const panel = btn.closest(".panel");
      const open = panel.dataset.open === "true";
      panel.dataset.open = open ? "false" : "true";
    });
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Camera feed fallback detection
  // ═══════════════════════════════════════════════════════════════════════
  cameraImg.addEventListener("error", () => { cameraFallback.style.display = "flex"; liveDot.style.opacity = .3; });
  cameraImg.addEventListener("load",  () => { cameraFallback.style.display = "none"; liveDot.style.opacity = 1; });
  cameraFallback.style.display = "flex";

  // ═══════════════════════════════════════════════════════════════════════
  // WebSocket
  // ═══════════════════════════════════════════════════════════════════════
  function setConn(state) {
    connPill.classList.remove("live", "err");
    if (state === "live") {
      connPill.classList.add("live");
      connText.textContent = "Live";
    } else if (state === "err") {
      connPill.classList.add("err");
      connText.textContent = "Offline";
    } else {
      connText.textContent = "Connecting\u2026";
    }
  }

  function renderVoiceStatus(v) {
    if (!v || !v.ts) {
      voicePill.classList.remove("live", "warn");
      voiceText.textContent = "Voice: idle";
      return;
    }
    const pct = Math.round((v.similarity || 0) * 100);
    voicePill.classList.remove("live", "warn");
    if (v.is_known) {
      voicePill.classList.add("live");
      voiceText.textContent = `Voice: ${v.name} (${pct}%)`;
    } else {
      voicePill.classList.add("warn");
      voiceText.textContent = `Voice: unknown (${pct}%)`;
    }
  }

  function applySnapshot(snap) {
    setOrbState(snap.orb_state || "idle");
    statusText.textContent = snap.status || "Monitoring\u2026";
    people.clear();
    (snap.tracks || []).forEach(t => people.set(t.track_id, t));
    renderPeople();
    eventsList.innerHTML = "";
    eventsEmpty.hidden = (snap.events || []).length > 0;
    (snap.events || []).forEach(ev => addEventRow(ev, false));
    renderVoiceStatus(snap.voice_status);
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onopen = () => setConn("live");
    ws.onclose = () => { setConn("err"); setTimeout(connect, 1500); };
    ws.onerror = () => ws.close();

    ws.onmessage = (evt) => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch { return; }

      switch (msg.type) {
        case "snapshot":
          applySnapshot(msg.data);
          break;
        case "orb":
          setOrbState(msg.state);
          break;
        case "status":
          setStatusText(msg.text);
          if (msg.orb_state) setOrbState(msg.orb_state);
          break;
        case "amplitude":
          targetAmplitude = msg.value;
          break;
        case "verdict":
          people.set(msg.data.track_id, msg.data);
          renderPeople();
          break;
        case "track_lost":
          if (people.has(msg.track_id)) {
            people.get(msg.track_id).status = "lost";
            people.get(msg.track_id).last_seen = Date.now() / 1000;
            renderPeople();
          }
          break;
        case "remove_track":
          people.delete(msg.track_id);
          renderPeople();
          break;
        case "clear_tracks":
          people.clear();
          renderPeople();
          break;
        case "event":
          addEventRow(msg.data, true);
          break;
        case "voice_status":
          renderVoiceStatus(msg.data);
          break;
      }
    };
  }

  // ═══════════════════════════════════════════════════════════════════════

  // ═══════════════════════════════════════════════════════════════════════
  // Controls — modem, emergency alert, shutdown
  // ═══════════════════════════════════════════════════════════════════════
  const modemDot            = document.getElementById("modemDot");
  const modemStatusText     = document.getElementById("modemStatusText");
  const modemPortInput      = document.getElementById("modemPortInput");
  const modemConnectBtn     = document.getElementById("modemConnectBtn");
  const modemDisconnectBtn  = document.getElementById("modemDisconnectBtn");
  const modemActionStatus   = document.getElementById("modemActionStatus");
  const alertMessageInput   = document.getElementById("alertMessageInput");
  const alertSimulateCheck  = document.getElementById("alertSimulateCheck");
  const sendAlertBtn        = document.getElementById("sendAlertBtn");
  const alertActionStatus   = document.getElementById("alertActionStatus");
  const shutdownBtn         = document.getElementById("shutdownBtn");

  async function refreshModemStatus() {
    try {
      const res = await fetch("/api/modem/status").then(r => r.json());
      if (res.connected) {
        modemDot.className = "modem-dot on";
        modemStatusText.textContent = `connected${res.port ? " (" + res.port + ")" : ""}`;
      } else {
        modemDot.className = "modem-dot off";
        modemStatusText.textContent = "not connected";
      }
    } catch {
      modemDot.className = "modem-dot off";
      modemStatusText.textContent = "unknown";
    }
  }

  modemConnectBtn.addEventListener("click", async () => {
    modemConnectBtn.disabled = true;
    modemActionStatus.textContent = "Connecting\u2026";
    modemActionStatus.className = "enroll-status";
    const port = modemPortInput.value.trim() || null;
    const res = await fetch("/api/modem/connect", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ port }),
    }).then(r => r.json()).catch(() => ({ ok: false, error: "Network error" }));
    modemConnectBtn.disabled = false;
    if (res.connected) {
      modemActionStatus.textContent = "Connected.";
      modemActionStatus.className = "enroll-status ok";
    } else {
      modemActionStatus.textContent = res.error || "Could not connect.";
      modemActionStatus.className = "enroll-status err";
    }
    refreshModemStatus();
  });

  modemDisconnectBtn.addEventListener("click", async () => {
    await fetch("/api/modem/disconnect", { method: "POST" });
    modemActionStatus.textContent = "Disconnected.";
    modemActionStatus.className = "enroll-status";
    refreshModemStatus();
  });

  sendAlertBtn.addEventListener("click", async () => {
    sendAlertBtn.disabled = true;
    alertActionStatus.textContent = "Sending\u2026";
    alertActionStatus.className = "enroll-status";
    const res = await fetch("/api/alert/emergency", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: alertMessageInput.value.trim() || "Test alert",
        simulate: alertSimulateCheck.checked,
      }),
    }).then(r => r.json()).catch(() => ({ ok: false, error: "Network error" }));
    sendAlertBtn.disabled = false;

    if (!res.ok) {
      alertActionStatus.textContent = res.error || "Failed.";
      alertActionStatus.className = "enroll-status err";
      return;
    }
    if (res.simulated) {
      const names = (res.result.would_text || []).join(", ") || "no one — add emergency contacts first";
      alertActionStatus.textContent = `Simulated — would text: ${names}`;
      alertActionStatus.className = "enroll-status ok";
    } else {
      const texted = (res.result.texted || []).join(", ") || "no one";
      alertActionStatus.textContent = `Sent — texted: ${texted}`;
      alertActionStatus.className = "enroll-status ok";
    }
  });

  shutdownBtn.addEventListener("click", async () => {
    if (!confirm("Shut down Nova? This stops the whole system.")) return;
    shutdownBtn.disabled = true;
    shutdownBtn.textContent = "Shutting down\u2026";
    await fetch("/api/shutdown", { method: "POST" }).catch(() => {});
  });

  refreshModemStatus();
  setInterval(refreshModemStatus, 10000);

  silenceAlarmBtn.addEventListener("click", async () => {
    silenceAlarmBtn.disabled = true;
    silenceAlarmBtn.textContent = "Silencing\u2026";
    await fetch("/api/alarm/silence", { method: "POST" }).catch(() => {});
    silenceAlarmBtn.disabled = false;
    silenceAlarmBtn.textContent = "Silence Alarm";
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Notification modes — three toggle switches
  // ═══════════════════════════════════════════════════════════════════════
  const toggleDigest        = document.getElementById("toggleDigest");
  const digestIntervalInput = document.getElementById("digestIntervalInput");
  const toggleUnknownText   = document.getElementById("toggleUnknownText");
  const toggleUnknownCall   = document.getElementById("toggleUnknownCall");
  const notifSettingsStatus = document.getElementById("notifSettingsStatus");

  let notifSettingsLoaded = false;

  async function loadNotificationSettings() {
    const res = await fetch("/api/settings/notifications").then(r => r.json()).catch(() => null);
    if (!res || !res.ok) return;
    const s = res.settings;
    toggleDigest.checked = !!s.log_digest_enabled;
    digestIntervalInput.value = s.log_digest_interval_minutes || 10;
    toggleUnknownText.checked = !!s.unknown_text_enabled;
    toggleUnknownCall.checked = !!s.unknown_call_enabled;
    notifSettingsLoaded = true;
  }

  async function pushNotificationSettings(patch) {
    notifSettingsStatus.textContent = "Saving\u2026";
    notifSettingsStatus.className = "enroll-status";
    const res = await fetch("/api/settings/notifications", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }).then(r => r.json()).catch(() => ({ ok: false }));
    if (res.ok) {
      notifSettingsStatus.textContent = "Saved.";
      notifSettingsStatus.className = "enroll-status ok";
    } else {
      notifSettingsStatus.textContent = res.error || "Could not save.";
      notifSettingsStatus.className = "enroll-status err";
    }
    clearTimeout(pushNotificationSettings._t);
    pushNotificationSettings._t = setTimeout(() => {
      notifSettingsStatus.textContent = "";
    }, 3000);
  }

  toggleDigest.addEventListener("change", () =>
    pushNotificationSettings({ log_digest_enabled: toggleDigest.checked }));
  toggleUnknownText.addEventListener("change", () =>
    pushNotificationSettings({ unknown_text_enabled: toggleUnknownText.checked }));
  toggleUnknownCall.addEventListener("change", () =>
    pushNotificationSettings({ unknown_call_enabled: toggleUnknownCall.checked }));
  digestIntervalInput.addEventListener("change", () => {
    const v = Math.max(1, Math.min(180, parseInt(digestIntervalInput.value, 10) || 10));
    digestIntervalInput.value = v;
    pushNotificationSettings({ log_digest_interval_minutes: v });
  });

  loadNotificationSettings();

  // ═══════════════════════════════════════════════════════════════════════
  // Emergency contacts — list (by priority), call, message, add
  // ═══════════════════════════════════════════════════════════════════════
  const contactsList    = document.getElementById("contactsList");
  const contactsEmpty   = document.getElementById("contactsEmpty");
  const contactsBadge   = document.getElementById("contactsBadge");
  const newContactName     = document.getElementById("newContactName");
  const newContactPhone    = document.getElementById("newContactPhone");
  const newContactRole     = document.getElementById("newContactRole");
  const newContactPriority = document.getElementById("newContactPriority");
  const addContactBtn      = document.getElementById("addContactBtn");
  const addContactStatus   = document.getElementById("addContactStatus");

  function renderContacts(contacts) {
    contactsBadge.textContent = contacts.length;
    contactsEmpty.hidden = contacts.length > 0;
    contactsList.innerHTML = contacts.map((c, i) => `
      <div class="contact-row" data-id="${c.id}">
        <div class="contact-priority">${i === 0 ? "\u2605" : c.priority}</div>
        <div class="contact-info">
          <div class="contact-name">${c.name}${c.role ? ` <span class="toggle-sub">(${c.role})</span>` : ""}</div>
          <div class="contact-meta">${c.phone}</div>
        </div>
        <div class="contact-actions">
          <button class="btn-small contact-call-btn" data-id="${c.id}">Call</button>
          <button class="btn-small btn-ghost contact-msg-toggle" data-id="${c.id}">Message</button>
          <button class="icon-btn contact-delete-btn" data-id="${c.id}" title="Remove">&times;</button>
        </div>
      </div>
      <div class="contact-msg-row" id="msg-row-${c.id}" hidden>
        <input type="text" class="train-path-input contact-msg-input" data-id="${c.id}"
               placeholder="Message to ${c.name}">
        <button class="btn-small btn-train contact-msg-send" data-id="${c.id}">Send</button>
      </div>
      <div class="enroll-status" id="contact-status-${c.id}" style="padding-left:35px;"></div>
    `).join("");
  }

  async function refreshContacts() {
    const res = await fetch("/api/contacts/emergency").then(r => r.json()).catch(() => null);
    if (res && res.ok) renderContacts(res.contacts || []);
  }

  function contactStatus(id, text, cls) {
    const el = document.getElementById(`contact-status-${id}`);
    if (!el) return;
    el.textContent = text;
    el.className = "enroll-status" + (cls ? " " + cls : "");
  }

  contactsList.addEventListener("click", async (e) => {
    const callBtn = e.target.closest(".contact-call-btn");
    const msgToggle = e.target.closest(".contact-msg-toggle");
    const sendBtn = e.target.closest(".contact-msg-send");
    const delBtn = e.target.closest(".contact-delete-btn");

    if (callBtn) {
      const id = callBtn.dataset.id;
      callBtn.disabled = true;
      contactStatus(id, "Calling\u2026");
      const res = await fetch(`/api/contacts/emergency/${id}/call`, { method: "POST" })
        .then(r => r.json()).catch(() => ({ ok: false, error: "Network error" }));
      callBtn.disabled = false;
      if (res.ok) {
        contactStatus(id, res.answered ? "Call answered." : "Not answered.", res.answered ? "ok" : "err");
      } else {
        contactStatus(id, res.error || "Call failed.", "err");
      }
      return;
    }

    if (msgToggle) {
      const id = msgToggle.dataset.id;
      const row = document.getElementById(`msg-row-${id}`);
      row.hidden = !row.hidden;
      if (!row.hidden) row.querySelector(".contact-msg-input").focus();
      return;
    }

    if (sendBtn) {
      const id = sendBtn.dataset.id;
      const input = contactsList.querySelector(`.contact-msg-input[data-id="${id}"]`);
      const message = input.value.trim();
      if (!message) { contactStatus(id, "Type a message first.", "err"); return; }
      sendBtn.disabled = true;
      contactStatus(id, "Sending\u2026");
      const res = await fetch(`/api/contacts/emergency/${id}/message`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      }).then(r => r.json()).catch(() => ({ ok: false, error: "Network error" }));
      sendBtn.disabled = false;
      if (res.ok && res.sent) {
        contactStatus(id, "Sent.", "ok");
        input.value = "";
      } else {
        contactStatus(id, (res && res.error) || "Message failed.", "err");
      }
      return;
    }

    if (delBtn) {
      const id = delBtn.dataset.id;
      if (!confirm("Remove this emergency contact?")) return;
      await fetch(`/api/contacts/emergency/${id}`, { method: "DELETE" });
      refreshContacts();
      return;
    }
  });

  addContactBtn.addEventListener("click", async () => {
    const name  = newContactName.value.trim();
    const phone = newContactPhone.value.trim();
    if (!name || !phone) {
      addContactStatus.textContent = "Name and phone are required.";
      addContactStatus.className = "enroll-status err";
      return;
    }
    addContactBtn.disabled = true;
    addContactStatus.textContent = "Adding\u2026";
    addContactStatus.className = "enroll-status";
    const res = await fetch("/api/contacts/emergency", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name, phone,
        role: newContactRole.value.trim(),
        priority: parseInt(newContactPriority.value, 10) || 1,
      }),
    }).then(r => r.json()).catch(() => ({ ok: false, error: "Network error" }));
    addContactBtn.disabled = false;
    if (res.ok) {
      addContactStatus.textContent = "Added.";
      addContactStatus.className = "enroll-status ok";
      newContactName.value = "";
      newContactPhone.value = "";
      newContactRole.value = "";
      newContactPriority.value = "1";
      renderContacts(res.contacts || []);
    } else {
      addContactStatus.textContent = res.error || "Could not add contact.";
      addContactStatus.className = "enroll-status err";
    }
  });

  refreshContacts();
  setInterval(refreshContacts, 30000);

  // Initial REST snapshot so the UI isn't empty before the socket opens
  fetch("/api/snapshot").then(r => r.json()).then(applySnapshot).catch(() => {});
  renderStatusList();
  connect();

  // periodically refresh "lost X sec ago" timers
  setInterval(renderPeople, 5000);
})();

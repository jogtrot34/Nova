"""
server.py

Nova web backend — FastAPI + WebSocket.

This is a DROP-IN REPLACEMENT for nova_ui.py's public interface.
Everywhere your existing code does:

    from nova_ui import NovaUI, Verdict, OrbState
    ui = NovaUI(mic_device=1)
    ...
    ui.run()                 # opened a Flet desktop window

you now do:

    from server import NovaWebUI as NovaUI, Verdict, OrbState
    ui = NovaUI(mic_device=1)
    ...
    ui.run()                 # starts the web server + serves the UI

Every call site in identify.py / main.py (push_verdict, set_status,
set_detected, set_identifying, set_speaking, remove_track,
clear_tracks) works completely unchanged. One new method is added,
trigger_wake_word(), which you call from voice_layer.py the moment
the "Nova" wake word is heard.

Run standalone for UI development (no camera/mic/db needed):
    python3 server.py --demo

Run for real, wired into main.py:
    python3 main.py               # after swapping the import above
"""

import argparse
import asyncio
import contextlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import uvicorn

# web/server.py can be run directly (`python3 web/server.py`) or imported
# as `web.server` from the project root — either way, make sure the
# project root (parent of this file's directory) is importable so
# `db.py`, `face_layer.py`, `voice_layer.py` etc. are always reachable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    _SD_AVAILABLE = False

# The DB and enrollment helpers are optional imports — the web UI should
# still boot in --demo mode even on a machine without dlib/resemblyzer
# installed. Real enrollment endpoints just report a clear error if these
# aren't available.
try:
    from db import NovaDB
    _DB_AVAILABLE = True
except Exception as e:
    _DB_AVAILABLE = False
    _DB_IMPORT_ERROR = str(e)

try:
    from face_layer import (encode_face_from_image, encode_face_from_frame,
                             ENCODING_JITTER_ENROLL,
                             enroll_person_faces_from_folder)
    _FACE_LAYER_AVAILABLE = True
except Exception as e:
    _FACE_LAYER_AVAILABLE = False
    _FACE_LAYER_IMPORT_ERROR = str(e)

try:
    from voice_layer import (record_audio, embed_audio, RECORD_SECONDS,
                              VOICE_DIR, SAMPLE_RATE,
                              enroll_person_voice_from_folder)
    _VOICE_LAYER_AVAILABLE = True
except Exception as e:
    _VOICE_LAYER_AVAILABLE = False
    _VOICE_LAYER_IMPORT_ERROR = str(e)

try:
    from contacts_db import ContactsDB
    _CONTACTS_AVAILABLE = True
except Exception as e:
    _CONTACTS_AVAILABLE = False
    _CONTACTS_IMPORT_ERROR = str(e)

try:
    from notifier import Notifier
    _NOTIFIER_AVAILABLE = True
except Exception as e:
    _NOTIFIER_AVAILABLE = False
    _NOTIFIER_IMPORT_ERROR = str(e)

try:
    from notification_settings import NotificationSettings
    _SETTINGS_AVAILABLE = True
except Exception as e:
    _SETTINGS_AVAILABLE = False
    _SETTINGS_IMPORT_ERROR = str(e)


STATIC_DIR = Path(__file__).parent / "static"

# ── Orb / state names (identical meaning to the old Flet OrbState) ────────────

class OrbState:
    IDLE        = "idle"        # cyan, calm breathing        — monitoring
    DETECTED    = "detected"    # orange, fast pulse          — person seen, not ID'd
    IDENTIFYING = "identifying" # yellow, rapid flicker       — running face/voice match
    SPEAKING    = "speaking"    # blue, active waveform       — Nova is talking
    CONFIRMED   = "confirmed"   # green, strong flash         — positive ID
    WAITING     = "waiting"     # amber, waiting pulse        — needs voice to confirm
    DENIED      = "denied"      # red, sharp pulse            — unknown / conflict
    WAKE        = "wake"        # violet, burst               — wake word "Nova" heard
    PIN_REQUIRED= "pin_required"# orange, steady               — borderline match, enter PIN
    SAFE_WORD   = "safe_word"   # orange, steady               — duress phrase armed
    ALARM       = "alarm"       # rapid multi-colour flash     — unknown-only room, sustained


# How long an unrecognised person must be alone in frame (no known
# person also present) before the intruder alarm fires.
ALARM_TRIGGER_SECONDS = 60.0
ALARM_SOUND_PATH = "assets/alarm/intruder_alarm.mp3"
# After a manual "Silence Alarm", don't let the same still-lingering
# stranger immediately re-trigger it — give a quiet window first.
ALARM_SNOOZE_SECONDS = 120


# How long a lost track stays visible (as "Lost sight") before it disappears
# from the Detected People panel entirely.
LOST_GRACE_SECONDS = 20.0

# Someone Nova actually confirmed gets a much longer grace period —
# assume they're still nearby for a while instead of immediately
# treating a momentary loss of view as "gone". Mirrors identify.py's
# own PRESENCE_COOLDOWN so the UI and the recognition logic agree.
CONFIRMED_PRESENCE_COOLDOWN = 30 * 60

# How long the WAKE orb state holds before reverting to IDLE
WAKE_HOLD_SECONDS = 2.5

MAX_EVENTS = 60


# ── Verdict (same shape identify.py already builds) ───────────────────────────

@dataclass
class Verdict:
    track_id:      int
    name:          str            = "Unknown"
    person_id:     Optional[int]  = None
    face_conf:      float         = 0.0
    voice_conf:     float         = 0.0
    combined_conf:  float         = 0.0
    method:         str           = ""
    access:         str           = "none"
    decision:       str           = "denied"
    top_desc:       str           = "not visible"
    bottom_desc:    str           = "not visible"
    gender:         str           = "unknown"
    conflict:       bool          = False
    needs_voice:    bool          = False
    pin_required:        bool           = False
    candidate_person_id: Optional[int]  = None
    candidate_name:      str            = ""
    runner_up_name:      str            = ""
    ambiguous:           bool           = False


@dataclass
class _TrackEntry:
    verdict:   Verdict
    status:    str = "active"      # active | tracking | lost
    last_seen: float = field(default_factory=time.time)
    lost_at:   Optional[float] = None


# ── Mic amplitude monitor (unchanged behaviour from nova_ui.py) ───────────────

class MicMonitor:
    def __init__(self, device: Optional[int] = None, samplerate: int = 44100,
                 blocksize: int = 512):
        self._amplitude  = 0.0
        self._running    = False
        self._lock       = threading.Lock()
        self._device     = device   # None = auto-detect a working device
        self._samplerate = samplerate
        self._blocksize  = blocksize

    def _callback(self, indata, frames, time_info, status):
        rms = float(np.sqrt(np.mean(indata ** 2)))
        normalised = min(rms * 12.0, 1.0)
        with self._lock:
            self._amplitude = self._amplitude * 0.7 + normalised * 0.3

    def start(self):
        if not _SD_AVAILABLE:
            return
        # Disabled by default — this opens its own independent mic
        # stream purely for the cosmetic waveform animation, which on
        # a raw ALSA hardware device (as opposed to the shareable
        # 'default'/pulse device) fights voice_layer.py's verification
        # stream for exclusive access. voice_layer.py's identification
        # is the only thing that should be touching the mic right now.
        # Re-enable once you're back on a shareable input device:
        #
        #     NOVA_ENABLE_MIC_WAVEFORM=1 python3 main.py
        if os.environ.get("NOVA_ENABLE_MIC_WAVEFORM") != "1":
            print("[MicMonitor] Disabled (mic reserved for voice_layer.py "
                  "verification) — waveform will stay flat. Set "
                  "NOVA_ENABLE_MIC_WAVEFORM=1 to turn it back on.")
            return
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _open_stream(self, device):
        return sd.InputStream(
            device=device, channels=1,
            samplerate=self._samplerate, blocksize=self._blocksize,
            callback=self._callback, dtype="float32",
        )

    def _run(self):
        device = self._device
        try:
            with self._open_stream(device):
                while self._running:
                    time.sleep(0.05)
            return
        except Exception as e:
            print(f"[MicMonitor] Could not open mic (device={device}): {e}")

        # The requested/default device didn't work — try auto-detecting
        # a device that actually has input channels before giving up.
        try:
            from audio_devices import configure_audio_device
            in_idx, _ = configure_audio_device()
            if in_idx is None or in_idx == device:
                print("[MicMonitor] No alternative input device found — "
                      "mic-reactive waveform will stay idle. Run "
                      "'python3 audio_devices.py' to see what's available.")
                return
            print(f"[MicMonitor] Retrying with auto-detected device={in_idx}")
            with self._open_stream(in_idx):
                self._device = in_idx
                while self._running:
                    time.sleep(0.05)
        except Exception as e:
            print(f"[MicMonitor] Auto-detected device also failed: {e}")
            print("[MicMonitor] Mic-reactive waveform will stay idle.")

    def stop(self):
        self._running = False

    @property
    def amplitude(self) -> float:
        with self._lock:
            return self._amplitude


# ── WebSocket connection manager (thread-safe broadcast) ──────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def _broadcast(self, message: dict):
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast(self, message: dict):
        """Safe to call from any thread (camera loop, voice thread, etc)."""
        if self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)
        except RuntimeError:
            pass


# ── NovaWebUI — same public surface as nova_ui.NovaUI ─────────────────────────

class NovaWebUI:

    def __init__(self, mic_device: Optional[int] = None):
        self._tracks:     dict[int, _TrackEntry] = {}
        self._status:      str  = "Monitoring..."
        self._orb_state:   str  = OrbState.IDLE
        self._events:      list[dict] = []
        self._lock         = threading.Lock()
        self._mic          = MicMonitor(device=mic_device)
        self._manager       = manager
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_lock     = threading.Lock()
        self._wake_timer:  Optional[threading.Timer] = None
        self._uvicorn_server = None
        self._voice        = None   # optional audio_responses.VoiceBank
        self._notifier      = None   # optional notifier.Notifier
        self._engine        = None   # optional identify.IdentifyEngine
        self.db              = NovaDB() if _DB_AVAILABLE else None
        self._voice_buffer: dict[int, list] = {}   # person_id -> [embeddings]
        self._voice_buffer_lock = threading.Lock()

        # ── Notification modes (text/call toggles) ─────────────────────────
        self._settings = NotificationSettings() if _SETTINGS_AVAILABLE else None
        self._id_log_lock = threading.Lock()
        self._id_log_buffer: list[dict] = []   # cleared each digest cycle
        self._digest_elapsed = 0.0
        # track_id -> {"since": ts, "texted": bool, "called": bool} —
        # lets unknown_text/unknown_call fire once per sighting instead
        # of spamming a text/call every frame a stranger is in view.
        self._unknown_notify_state: dict[int, dict] = {}

        # ── Intruder alarm ──────────────────────────────────────────────
        self._alarm_active = False
        self._alarm_lock = threading.Lock()
        self._alarm_sound_cache = None
        self._alarm_snoozed_until = 0.0

        # ── Live voice-ID status — independent of face tracks ───────────
        # (see push_voice_status) so the dashboard shows voice_layer.py
        # working even when no face is currently being tracked.
        self._voice_status = {"name": None, "similarity": 0.0,
                              "is_known": False, "ts": 0}

        app.state.ui = self   # let FastAPI routes reach this instance

    def set_engine(self, engine):
        """Attach the running IdentifyEngine so people-management
        endpoints can tell it to reload face/voice data live."""
        self._engine = engine

    def _reload_engine(self):
        if self._engine:
            try:
                self._engine.reload_people()
            except Exception as e:
                print(f"[NovaWebUI] Engine reload failed: {e}")

    def set_voice_bank(self, voice_bank):
        """Attach an audio_responses.VoiceBank so trigger_wake_word() etc
        can play the matching pre-recorded line automatically."""
        self._voice = voice_bank

    def set_notifier(self, notifier):
        """Attach a notifier.Notifier so the web UI's alert endpoints
        (/api/alert/emergency, /api/alert/person/<id>) work."""
        self._notifier = notifier
        app.state.notifier = notifier

    # ── Notification modes: digest / unknown-text / unknown-call ─────────────

    def _top_emergency_contact(self) -> Optional[dict]:
        """Contact #1 — lowest priority number, same convention
        safe_word.py already uses for 'the one alert goes to me'."""
        if not _CONTACTS_AVAILABLE:
            return None
        try:
            contacts = ContactsDB().list_emergency_contacts()
        except Exception:
            return None
        return contacts[0] if contacts else None

    def _send_text(self, phone: str, message: str) -> bool:
        if not self._notifier:
            return False
        try:
            return bool(self._notifier.text(phone, message))
        except Exception as e:
            print(f"[NovaWebUI] Text failed: {e}")
            return False

    def _place_call(self, phone: str) -> bool:
        if not self._notifier:
            return False
        try:
            return bool(self._notifier.call(phone))
        except Exception as e:
            print(f"[NovaWebUI] Call failed: {e}")
            return False

    def record_identification(self, name: str, decision: str):
        """Called from push_verdict for every real (non-Unknown)
        identity match — feeds the periodic 'identification logs'
        digest text. Cheap: just an in-memory list, flushed by
        _digest_loop()."""
        with self._id_log_lock:
            self._id_log_buffer.append({
                "name": name, "decision": decision,
                "ts": time.strftime("%H:%M:%S"),
            })

    def notify_unknown_detected(self, track_id: int, desc: str):
        """Called once per sighting (not once per frame) the moment an
        unrecognised person is confirmed in frame. Texts emergency
        contact #1 if that mode is switched on."""
        state = self._unknown_notify_state.setdefault(
            track_id, {"since": time.time(), "texted": False, "called": False})

        settings = self._settings.get_all() if self._settings else {}
        if not settings.get("unknown_text_enabled") or state["texted"]:
            return
        contact = self._top_emergency_contact()
        if not contact:
            self._log_event("info", "Unknown-person text skipped — "
                             "no emergency contact on file", desc)
            state["texted"] = True
            return
        message = f"[Nova] Unknown person detected — {desc}."
        sent = self._send_text(contact["phone"], message)
        state["texted"] = True
        self._log_event("info",
                         "Unknown-person text sent" if sent else
                         "Unknown-person text NOT sent (no modem connected)",
                         f"{contact['name']} — {message}")

    def notify_unknown_persists(self, track_id: int, desc: str):
        """Called every frame an unrecognised person is still in view.
        Escalates to a call to emergency contact #1 once that person
        has stuck around past unknown_call_after_seconds — never on
        the very first frame, so it doesn't dial out on every passing
        stranger."""
        state = self._unknown_notify_state.get(track_id)
        if state is None:
            state = self._unknown_notify_state.setdefault(
                track_id, {"since": time.time(), "texted": False, "called": False})

        settings = self._settings.get_all() if self._settings else {}
        if not settings.get("unknown_call_enabled") or state["called"]:
            return
        threshold = settings.get("unknown_call_after_seconds", 15)
        if time.time() - state["since"] < threshold:
            return
        contact = self._top_emergency_contact()
        if not contact:
            self._log_event("info", "Unknown-person call skipped — "
                             "no emergency contact on file", desc)
            state["called"] = True
            return
        called = self._place_call(contact["phone"])
        state["called"] = True
        self._log_event("info",
                         "Unknown-person call placed" if called else
                         "Unknown-person call NOT placed (no modem connected "
                         "or not answered)",
                         f"{contact['name']} — {desc}")

    def clear_unknown_notify_state(self, track_id: int):
        self._unknown_notify_state.pop(track_id, None)

    def notify_pin_failure(self, name: str):
        """A person who resembles someone enrolled entered the wrong
        PIN. Always attempts to alert every emergency contact (not
        gated behind a toggle — this is a specific access-control
        event, not routine chatter) so a human can decide whether to
        let them in anyway."""
        message = (f"A person who looks like {name} tried to gain access "
                   f"to Nova but failed PIN confirmation. Should I let "
                   f"them in?")
        if self._notifier:
            try:
                result = self._notifier.alert_emergency_contacts(message)
                self._log_event("info", "PIN failure — emergency contacts alerted",
                                message)
                return result
            except Exception as e:
                self._log_event("info", "PIN failure alert error", str(e))
                return {"error": str(e)}
        else:
            self._log_event("info", "PIN failure (no modem connected — not sent)",
                            message)
            return {"error": "No modem connected"}

    # ── Intruder alarm — unknown-only room, sustained ─────────────────────

    def _load_alarm_sound(self):
        if self._alarm_sound_cache is None:
            import librosa
            data, sr = librosa.load(ALARM_SOUND_PATH, sr=None, mono=True)
            self._alarm_sound_cache = (data, sr)
        return self._alarm_sound_cache

    def _alarm_audio_loop(self):
        try:
            data, sr = self._load_alarm_sound()
        except Exception as e:
            print(f"[NovaWebUI] Could not load alarm sound: {e}")
            self._log_event("info", "Alarm sound could not be loaded", str(e))
            return
        try:
            import sounddevice as sd
        except Exception as e:
            print(f"[NovaWebUI] sounddevice unavailable for alarm: {e}")
            return
        while self._alarm_active:
            try:
                sd.play(data, samplerate=sr)
                sd.wait()
            except Exception as e:
                print(f"[NovaWebUI] Alarm playback error: {e}")
                time.sleep(1.0)

    def _notify_alarm(self):
        message = ("[Nova] ALARM: an unidentified person has been alone in "
                   "view for over a minute with no recognised person "
                   "present. Calling now.")
        if self._notifier:
            try:
                self._notifier.alert_emergency_contacts(message, also_call=True)
                self._log_event("info", "Alarm — contacts texted, top "
                                "contact called", message)
            except Exception as e:
                self._log_event("info", "Alarm notify error", str(e))
        else:
            self._log_event("info", "Alarm (no modem connected — "
                            "text/call not sent)", message)

    def trigger_alarm(self):
        """Idempotent — safe to call every frame while the condition
        holds. No-ops if already alarming or still within the
        post-silence snooze window."""
        with self._alarm_lock:
            if self._alarm_active:
                return
            if time.time() < self._alarm_snoozed_until:
                return
            self._alarm_active = True
        self._log_event("info", "ALARM — unidentified person alone in "
                        "view for over a minute", "")
        self.set_status("ALARM \u2014 unidentified person detected", OrbState.ALARM)
        threading.Thread(target=self._alarm_audio_loop, daemon=True).start()
        threading.Thread(target=self._notify_alarm, daemon=True).start()

    def clear_alarm(self):
        """Idempotent — safe to call every frame while the condition
        does not hold."""
        with self._alarm_lock:
            if not self._alarm_active:
                return
            self._alarm_active = False
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._log_event("info", "Alarm cleared", "")
        self.set_status("Monitoring\u2026", OrbState.IDLE)

    def silence_alarm(self, snooze_seconds: int = ALARM_SNOOZE_SECONDS):
        """Manual override from the UI — stops the siren/flash right
        away and holds off re-triggering for `snooze_seconds` even if
        the same stranger is still standing there."""
        self._alarm_snoozed_until = time.time() + snooze_seconds
        self.clear_alarm()

    def _digest_loop(self):
        """Background thread: every log_digest_interval_minutes, if
        log_digest_enabled and anything was logged since last time,
        texts emergency contact #1 a one-line summary. Sends nothing
        (and doesn't even check the modem) if the buffer is empty —
        no digest just to say 'nothing happened'."""
        if not self._settings:
            return
        TICK = 15.0   # seconds between checks — lets a live interval
                      # change take effect without waiting out a stale
                      # long interval first
        while True:
            time.sleep(TICK)
            self._digest_elapsed += TICK
            settings = self._settings.get_all()
            interval_seconds = max(60, settings.get("log_digest_interval_minutes", 10) * 60)
            if self._digest_elapsed < interval_seconds:
                continue
            self._digest_elapsed = 0.0
            if not settings.get("log_digest_enabled"):
                with self._id_log_lock:
                    self._id_log_buffer.clear()
                continue

            with self._id_log_lock:
                entries = list(self._id_log_buffer)
                self._id_log_buffer.clear()
            if not entries:
                continue   # only if any — nothing happened, send nothing

            contact = self._top_emergency_contact()
            if not contact:
                self._log_event("info", "Identification digest skipped — "
                                "no emergency contact on file", "")
                continue

            lines = [f"{e['ts']} {e['name']} ({e['decision']})" for e in entries[-10:]]
            more = f" (+{len(entries)-10} more)" if len(entries) > 10 else ""
            message = "[Nova] Identification log — " + "; ".join(lines) + more
            sent = self._send_text(contact["phone"], message)
            self._log_event("info",
                            "Identification digest sent" if sent else
                            "Identification digest NOT sent (no modem connected)",
                            f"{contact['name']} — {len(entries)} entr{'y' if len(entries)==1 else 'ies'}")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _log_event(self, kind: str, title: str, subtitle: str = ""):
        entry = {
            "ts": time.strftime("%H:%M:%S"),
            "kind": kind,
            "title": title,
            "subtitle": subtitle,
        }
        self._events.append(entry)
        if len(self._events) > MAX_EVENTS:
            self._events = self._events[-MAX_EVENTS:]
        self._manager.broadcast({"type": "event", "data": entry})

    def _set_orb(self, state: str):
        self._orb_state = state
        self._manager.broadcast({"type": "orb", "state": state})

    # ── Public API (matches nova_ui.NovaUI exactly) ──────────────────────────

    def push_voice_status(self, name: str, similarity: float, is_known: bool):
        """Called directly by VoiceLayer's background thread (via
        identify.py's _on_voice_result) every time it has a fresh
        result — completely independent of face tracks, so the "Voice"
        pill in the topbar updates live even when nobody's face is
        currently being tracked (bad angle, no camera, etc). This is
        what makes voice recognition visibly connected to the UI."""
        was_known_before = self._voice_status.get("is_known", False)
        self._voice_status = {
            "name": name if is_known else "Unknown",
            "similarity": round(float(similarity), 3),
            "is_known": bool(is_known),
            "ts": time.time(),
        }
        self._manager.broadcast({"type": "voice_status",
                                  "data": self._voice_status})
        # Only log the moment it flips TO a known match — logging every
        # "Unknown" sample (which fires every few seconds) would flood
        # Recent Activity with noise.
        if is_known and not was_known_before:
            self._log_event("voice", "Voice matched",
                            f"{name} \u2014 {similarity:.0%} similarity")

    def push_verdict(self, verdict: Verdict):
        with self._lock:
            entry = self._tracks.get(verdict.track_id)
            was_new = entry is None
            if entry is None:
                entry = _TrackEntry(verdict=verdict)
                self._tracks[verdict.track_id] = entry
            entry.verdict   = verdict
            entry.status    = "active"
            entry.last_seen = time.time()
            entry.lost_at   = None

            if verdict.conflict:
                self._orb_state = OrbState.DENIED
            elif verdict.needs_voice:
                self._orb_state = OrbState.WAITING
            elif verdict.decision == "granted":
                self._orb_state = OrbState.CONFIRMED
            else:
                self._orb_state = OrbState.DENIED

        if was_new:
            self._log_event("face", "Face detected",
                             f"Track {verdict.track_id:02d} — tracking")
        if verdict.decision == "granted" and verdict.name != "Unknown":
            self._log_event("identity", "Identity matched", verdict.name)
            self.record_identification(verdict.name, verdict.decision)
        if verdict.name != "Unknown":
            # Resolved (either granted or a denied-but-named case) —
            # a later re-appearance as a genuine stranger should be
            # able to re-trigger unknown_text/unknown_call.
            self.clear_unknown_notify_state(verdict.track_id)
        if verdict.needs_voice:
            self._log_event("voice", "Voice verification requested",
                             f"Track {verdict.track_id:02d}")

        self._manager.broadcast({"type": "orb", "state": self._orb_state})
        self._manager.broadcast({"type": "verdict",
                                  "data": self._track_payload(entry)})

    def set_detected(self):
        self._set_orb(OrbState.DETECTED)

    def set_identifying(self):
        self._set_orb(OrbState.IDENTIFYING)

    def set_speaking(self, speaking: bool):
        self._set_orb(OrbState.SPEAKING if speaking else OrbState.IDLE)

    def trigger_wake_word(self):
        """Call this the instant the wake word 'Nova' is detected."""
        self._log_event("wake", "Wake word detected", '"Nova"')
        self._set_orb(OrbState.WAKE)
        self.set_status("How can I help you?")
        if self._voice:
            self._voice.play("wake_ack", text_if_missing="Yes? I'm listening.",
                              blocking=False)
        if self._wake_timer:
            self._wake_timer.cancel()
        self._wake_timer = threading.Timer(WAKE_HOLD_SECONDS,
                                            lambda: self._set_orb(OrbState.IDLE))
        self._wake_timer.daemon = True
        self._wake_timer.start()

    def remove_track(self, track_id: int):
        self.clear_unknown_notify_state(track_id)
        with self._lock:
            entry = self._tracks.get(track_id)
            if entry is None:
                return
            entry.status  = "lost"
            entry.lost_at = time.time()
            was_confirmed = entry.verdict.decision == "granted"
            if not any(t.status == "active" for t in self._tracks.values()):
                self._orb_state = OrbState.IDLE

        self._manager.broadcast({"type": "orb", "state": self._orb_state})
        self._manager.broadcast({"type": "track_lost",
                                  "track_id": track_id})
        if was_confirmed:
            self._log_event("info", "Lost view of confirmed person — "
                             "assuming still nearby",
                             f"Track {track_id:02d}")
        else:
            self._log_event("info", "Lost sight of person",
                             f"Track {track_id:02d}")

        grace = CONFIRMED_PRESENCE_COOLDOWN if was_confirmed else LOST_GRACE_SECONDS

        def _purge():
            time.sleep(grace)
            with self._lock:
                cur = self._tracks.get(track_id)
                if cur and cur.status == "lost":
                    del self._tracks[track_id]
            self._manager.broadcast({"type": "remove_track",
                                      "track_id": track_id})
        threading.Thread(target=_purge, daemon=True).start()

    def clear_tracks(self):
        with self._lock:
            self._tracks.clear()
            self._orb_state = OrbState.IDLE
        self._manager.broadcast({"type": "clear_tracks"})
        self._manager.broadcast({"type": "orb", "state": OrbState.IDLE})

    def set_status(self, text: str, orb_state: Optional[str] = None):
        self._status = text
        if orb_state:
            self._orb_state = orb_state
        self._manager.broadcast({"type": "status", "text": text,
                                  "orb_state": self._orb_state})

    def push_frame(self, frame_bgr: np.ndarray):
        """Optional: feed camera frames in for the live Camera Feed panel."""
        if not _CV2_AVAILABLE:
            return
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            with self._jpeg_lock:
                self._latest_jpeg = buf.tobytes()

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_jpeg

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Blocking call — starts the web server (replaces ft.app)."""
        self._mic.start()
        threading.Thread(target=self._amplitude_loop, daemon=True).start()
        threading.Thread(target=self._digest_loop, daemon=True).start()
        print(f"[NovaWebUI] Serving on http://{host}:{port}")
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self._uvicorn_server = uvicorn.Server(config)
        self._uvicorn_server.run()

    def request_shutdown(self):
        """Gracefully stops the web server (and, since run() is what's
        blocking main.py/server.py's __main__, the whole process)."""
        if getattr(self, "_uvicorn_server", None):
            self._uvicorn_server.should_exit = True
        else:
            os._exit(0)

    # ── snapshot / payload builders ───────────────────────────────────────────

    def _track_payload(self, entry: _TrackEntry) -> dict:
        d = asdict(entry.verdict)
        d["status"] = entry.status
        d["last_seen"] = entry.last_seen
        return d

    def get_snapshot(self) -> dict:
        with self._lock:
            tracks = [self._track_payload(e) for e in self._tracks.values()]
        return {
            "status": self._status,
            "orb_state": self._orb_state,
            "tracks": tracks,
            "events": list(reversed(self._events)),
            "alarm_active": self._alarm_active,
            "voice_status": self._voice_status,
        }

    def _amplitude_loop(self):
        while True:
            time.sleep(0.05)
            self._manager.broadcast({"type": "amplitude",
                                      "value": round(self._mic.amplitude, 4)})


# ── FastAPI app ─────────────────────────────────────────────────────────────

manager = ConnectionManager()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    manager.bind_loop(asyncio.get_running_loop())
    yield


app = FastAPI(title="Nova Security Intelligence", lifespan=_lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/manage")
async def manage_page():
    return FileResponse(STATIC_DIR / "manage.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/snapshot")
async def snapshot():
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    return ui.get_snapshot() if ui else {"status": "starting", "orb_state": "idle",
                                          "tracks": [], "events": []}


@app.get("/api/stream")
async def camera_stream():
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)

    def gen():
        boundary = b"--frame"
        while True:
            frame = ui.get_latest_jpeg() if ui else None
            if frame:
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.066)   # ~15fps

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/camera/status")
async def camera_status():
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    has_signal = bool(ui and ui.get_latest_jpeg())
    return {"ok": True, "has_signal": has_signal}


# ── Shutdown ─────────────────────────────────────────────────────────────

@app.post("/api/shutdown")
async def shutdown():
    """Stops the whole process, cleanly. Called from the dashboard's
    shutdown button — just sends it, no confirmation round-trip needed
    server-side (the button itself should confirm if you want that)."""
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    if ui:
        ui._log_event("info", "Shutdown requested from the UI", "")

    def _do_shutdown():
        time.sleep(0.3)   # let the HTTP response actually flush first
        if ui:
            ui.request_shutdown()
        else:
            os._exit(0)

    threading.Thread(target=_do_shutdown, daemon=True).start()
    return {"ok": True, "message": "Shutting down..."}


# ── Modem configuration ──────────────────────────────────────────────────

@app.get("/api/modem/status")
async def modem_status():
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    notifier = getattr(app.state, "notifier", None)
    if not notifier:
        return {"ok": True, "connected": False, "port": None}
    modem = getattr(notifier, "modem", None)
    connected = bool(getattr(modem, "_connected", False))
    port = getattr(modem, "_port", None) or getattr(modem, "port", None)
    return {"ok": True, "connected": connected, "port": port}


@app.post("/api/modem/connect")
async def modem_connect(payload: dict):
    """Body: {"port": "/dev/ttyACM0"} — omit/null to auto-detect.
    Connects (or reconnects) the modem used for emergency calls/SMS,
    without needing to restart Nova."""
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    port = payload.get("port") or None

    try:
        from emergency_dial import EmergencyDialer
    except Exception as e:
        return {"ok": False, "connected": False,
                "error": f"emergency_dial unavailable: {e}"}
    if not _NOTIFIER_AVAILABLE:
        return {"ok": False, "connected": False,
                "error": f"notifier unavailable: {_NOTIFIER_IMPORT_ERROR}"}
    if not _CONTACTS_AVAILABLE:
        return {"ok": False, "connected": False,
                "error": f"contacts_db unavailable: {_CONTACTS_IMPORT_ERROR}"}

    def _connect():
        dialer = EmergencyDialer(port=port)
        ok = dialer.connect()
        return dialer, ok

    dialer, ok = await run_in_threadpool(_connect)

    if not ok:
        if ui:
            ui._log_event("info", "Modem connect failed",
                          port or "(auto-detect)")
        return {"ok": True, "connected": False,
                "error": "Could not connect to the modem — check it's "
                         "plugged in and the port is right."}

    notifier = Notifier(dialer, ContactsDB())
    if ui:
        ui.set_notifier(notifier)
        ui._log_event("info", "Modem connected", dialer._port or "auto-detected")
    return {"ok": True, "connected": True, "port": dialer._port}


@app.post("/api/modem/disconnect")
async def modem_disconnect():
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    notifier = getattr(app.state, "notifier", None)
    if notifier and hasattr(notifier.modem, "disconnect"):
        await run_in_threadpool(notifier.modem.disconnect)
    app.state.notifier = None
    if ui:
        ui._notifier = None
        ui._log_event("info", "Modem disconnected", "")
    return {"ok": True}


@app.post("/api/alert/emergency")
async def alert_emergency(payload: dict):
    """Body: {"message": "...", "call": false, "play_audio_path": null,
    "simulate": false}. Texts every emergency contact; also rings the
    top-priority one if "call" is true. simulate=true (or no modem
    attached at all) previews who *would* be contacted without
    actually sending anything — safe to click during a demo."""
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    notifier = getattr(app.state, "notifier", None)
    message   = payload.get("message", "Nova security alert triggered manually.")
    also_call = bool(payload.get("call", False))
    play_path = payload.get("play_audio_path")
    simulate  = bool(payload.get("simulate", False)) or not notifier

    if simulate:
        from contacts_db import ContactsDB
        contacts = ContactsDB().list_emergency_contacts()
        if ui:
            ui._log_event("info", "Emergency alert SIMULATED (not actually sent)",
                          f"{message} — {len(contacts)} contact(s)")
        return {"ok": True, "simulated": True,
                "result": {
                    "would_text": [c["name"] for c in contacts],
                    "would_call": [contacts[0]["name"]] if also_call and contacts else [],
                }}

    result = notifier.alert_emergency_contacts(message, also_call=also_call,
                                                play_audio_path=play_path)
    if ui:
        ui._log_event("info", "Emergency contacts alerted", message)
    return {"ok": True, "simulated": False, "result": result}


@app.post("/api/alert/person/{person_id}")
async def alert_person(person_id: int, payload: dict):
    """Body: {"message": "...", "call": false, "play_audio_path": null,
    "simulate": false}. Texts (and optionally calls) that specific
    person's own contacts (e.g. next of kin)."""
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    notifier = getattr(app.state, "notifier", None)
    message   = payload.get("message", "Nova is trying to reach you.")
    also_call = bool(payload.get("call", False))
    play_path = payload.get("play_audio_path")
    simulate  = bool(payload.get("simulate", False)) or not notifier

    if simulate:
        from contacts_db import ContactsDB
        contacts = ContactsDB().list_person_contacts(person_id)
        if ui:
            ui._log_event("info", "Contact alert SIMULATED (not actually sent)",
                          f"person {person_id} — {message}")
        return {"ok": True, "simulated": True,
                "result": {"would_text": [c["name"] for c in contacts]}}

    result = notifier.notify_person_contacts(person_id, message, also_call=also_call,
                                              play_audio_path=play_path)
    if ui:
        ui._log_event("info", f"Contacts notified for person {person_id}", message)
    return {"ok": True, "simulated": False, "result": result}


# ── Emergency contacts — list, add, remove, call, message ─────────────────

def _require_contacts():
    if not _CONTACTS_AVAILABLE:
        return None, {"ok": False, "error": f"contacts_db unavailable: "
                                              f"{_CONTACTS_IMPORT_ERROR}"}
    return ContactsDB(), None


@app.get("/api/contacts/emergency")
async def list_emergency_contacts_route():
    """Sorted by priority ascending (1 = called/texted first) — same
    ordering contacts_db.py's SQL query already guarantees."""
    contacts, err = _require_contacts()
    if err:
        return err
    return {"ok": True, "contacts": contacts.list_emergency_contacts()}


@app.post("/api/contacts/emergency")
async def add_emergency_contact_route(payload: dict):
    contacts, err = _require_contacts()
    if err:
        return err
    name  = (payload.get("name") or "").strip()
    phone = (payload.get("phone") or "").strip()
    if not name or not phone:
        return {"ok": False, "error": "Name and phone are required."}
    role     = (payload.get("role") or "").strip()
    try:
        priority = int(payload.get("priority", 1))
    except (TypeError, ValueError):
        priority = 1
    cid = contacts.add_emergency_contact(name, phone, role=role, priority=priority)
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    if ui:
        ui._log_event("info", "Emergency contact added", f"{name} — {phone}")
    return {"ok": True, "id": cid,
            "contacts": contacts.list_emergency_contacts()}


@app.delete("/api/contacts/emergency/{contact_id}")
async def delete_emergency_contact_route(contact_id: int):
    contacts, err = _require_contacts()
    if err:
        return err
    contacts.remove_emergency_contact(contact_id)
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    if ui:
        ui._log_event("info", "Emergency contact removed", f"id={contact_id}")
    return {"ok": True, "contacts": contacts.list_emergency_contacts()}


def _find_emergency_contact(contacts: "ContactsDB", contact_id: int) -> Optional[dict]:
    for c in contacts.list_emergency_contacts():
        if c["id"] == contact_id:
            return c
    return None


@app.post("/api/contacts/emergency/{contact_id}/call")
async def call_emergency_contact_route(contact_id: int):
    """Places a real call through the connected modem. If no modem is
    attached, reports that plainly instead of pretending it dialled."""
    contacts, err = _require_contacts()
    if err:
        return err
    contact = _find_emergency_contact(contacts, contact_id)
    if not contact:
        return {"ok": False, "error": "No such contact"}

    notifier = getattr(app.state, "notifier", None)
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    if not notifier:
        return {"ok": False, "error": "No modem connected — connect one "
                                       "from the Controls panel first."}

    def _do_call():
        return notifier.call(contact["phone"])

    answered = await run_in_threadpool(_do_call)
    if ui:
        ui._log_event("info",
                      "Call answered" if answered else "Call not answered",
                      f"{contact['name']} — {contact['phone']}")
    return {"ok": True, "answered": bool(answered)}


@app.post("/api/contacts/emergency/{contact_id}/message")
async def message_emergency_contact_route(contact_id: int, payload: dict):
    """Body: {"message": "..."}. Sends a one-off SMS to this specific
    contact, independent of the broadcast Emergency Alert button."""
    contacts, err = _require_contacts()
    if err:
        return err
    contact = _find_emergency_contact(contacts, contact_id)
    if not contact:
        return {"ok": False, "error": "No such contact"}

    message = (payload.get("message") or "").strip()
    if not message:
        return {"ok": False, "error": "Message text is required."}

    notifier = getattr(app.state, "notifier", None)
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    if not notifier:
        return {"ok": False, "error": "No modem connected — connect one "
                                       "from the Controls panel first."}

    def _do_text():
        return notifier.text(contact["phone"], message)

    sent = await run_in_threadpool(_do_text)
    if ui:
        ui._log_event("info", "Message sent" if sent else "Message failed",
                      f"{contact['name']} — {message}")
    return {"ok": True, "sent": bool(sent)}


# ── Notification modes — the three toggle switches ─────────────────────────

@app.get("/api/settings/notifications")
async def get_notification_settings():
    if not _SETTINGS_AVAILABLE:
        return {"ok": False, "error": f"notification_settings unavailable: "
                                       f"{_SETTINGS_IMPORT_ERROR}"}
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    settings = ui._settings if (ui and ui._settings) else NotificationSettings()
    return {"ok": True, "settings": settings.get_all()}


@app.post("/api/settings/notifications")
async def update_notification_settings(payload: dict):
    """Body: any subset of {log_digest_enabled, log_digest_interval_minutes,
    unknown_text_enabled, unknown_call_enabled, unknown_call_after_seconds}.
    Takes effect immediately — the digest thread re-reads settings every
    tick, no restart needed."""
    if not _SETTINGS_AVAILABLE:
        return {"ok": False, "error": f"notification_settings unavailable: "
                                       f"{_SETTINGS_IMPORT_ERROR}"}
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    settings = ui._settings if (ui and ui._settings) else NotificationSettings()
    updated = settings.update(payload)
    if ui:
        ui._log_event("info", "Notification settings updated",
                      ", ".join(f"{k}={v}" for k, v in payload.items()))
    return {"ok": True, "settings": updated}


# ── Intruder alarm ───────────────────────────────────────────────────────

@app.post("/api/alarm/silence")
async def silence_alarm_route():
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    if not ui:
        return {"ok": False, "error": "UI not ready"}
    ui.silence_alarm()
    return {"ok": True}


def _person_public(p: dict) -> dict:
    """Trim a DB row to what the UI needs, without dumping raw
    face-encoding / voiceprint-path / pin-hash blobs into every list
    response."""
    return {
        "id":           p["id"],
        "first_name":   p["first_name"],
        "last_name":    p["last_name"],
        "role":         p["role"],
        "access_level": p["access_level"],
        "notes":        p["notes"],
        "has_face":     bool(p["face_encodings"]),
        "has_voice":    bool(p["voiceprint_path"]),
        "has_pin":      bool(p["pin_hash"]) if "pin_hash" in p.keys() else False,
        "registered_at":p["registered_at"],
    }


def _require_db():
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    if not ui or not ui.db:
        return None, {"ok": False, "error": "Database not available "
                       + (f"({_DB_IMPORT_ERROR})" if not _DB_AVAILABLE else "")}
    return ui, None


# ── People CRUD ────────────────────────────────────────────────────────────

@app.get("/api/people")
async def list_people():
    ui, err = _require_db()
    if err:
        return err
    people = ui.db.list_people()
    return {"ok": True, "people": [_person_public(p) for p in people]}


@app.post("/api/people")
async def add_person(payload: dict):
    ui, err = _require_db()
    if err:
        return err
    first = (payload.get("first_name") or "").strip()
    last  = (payload.get("last_name") or "").strip()
    if not first or not last:
        return {"ok": False, "error": "first_name and last_name are required"}

    pid = ui.db.add_person(
        first, last,
        role=payload.get("role", "visitor"),
        access_level=payload.get("access_level", "none"),
        notes=payload.get("notes", ""),
    )
    ui._reload_engine()
    ui._log_event("identity", "Person added", f"{first} {last}")
    return {"ok": True, "person": _person_public(ui.db.get_person(pid))}


@app.patch("/api/people/{person_id}")
async def update_person(person_id: int, payload: dict):
    ui, err = _require_db()
    if err:
        return err
    if not ui.db.get_person(person_id):
        return {"ok": False, "error": "No such person"}
    ui.db.update_person(person_id, **payload)
    ui._reload_engine()
    return {"ok": True, "person": _person_public(ui.db.get_person(person_id))}


@app.delete("/api/people/{person_id}")
async def delete_person(person_id: int):
    ui, err = _require_db()
    if err:
        return err
    person = ui.db.get_person(person_id)
    if not person:
        return {"ok": False, "error": "No such person"}
    ui.db.delete_person(person_id)
    with ui._voice_buffer_lock:
        ui._voice_buffer.pop(person_id, None)
    ui._reload_engine()
    ui._log_event("info", "Person removed",
                  f"{person['first_name']} {person['last_name']}")
    return {"ok": True}


# ── PIN — fallback confirmation for borderline face/voice matches ──────────

@app.post("/api/people/{person_id}/pin")
async def set_person_pin(person_id: int, payload: dict):
    """Body: {"pin": "1234"}. Every enrolled person sets their own —
    used only when face/voice land in the uncertain middle ground, not
    as a bypass for a total stranger."""
    ui, err = _require_db()
    if err:
        return err
    person = ui.db.get_person(person_id)
    if not person:
        return {"ok": False, "error": "No such person"}
    pin = (payload.get("pin") or "").strip()
    if not pin or not pin.isdigit() or not (4 <= len(pin) <= 8):
        return {"ok": False, "error": "PIN must be 4-8 digits"}
    ui.db.set_person_pin(person_id, pin)
    return {"ok": True}


@app.delete("/api/people/{person_id}/pin")
async def remove_person_pin(person_id: int):
    ui, err = _require_db()
    if err:
        return err
    if not ui.db.get_person(person_id):
        return {"ok": False, "error": "No such person"}
    ui.db.remove_person_pin(person_id)
    return {"ok": True}


@app.post("/api/verify-pin")
async def verify_pin(payload: dict):
    """Body: {"track_id": 3, "person_id": 7, "pin": "1234"}. Called from
    the dashboard's PIN prompt. Only works against the track's own
    candidate — you can't PIN-confirm your way into being someone the
    camera isn't even suggesting you might be. Wrong PIN or no PIN set
    is a denial with no further fallback."""
    ui, err = _require_db()
    if err:
        return err
    track_id  = payload.get("track_id")
    person_id = payload.get("person_id")
    pin       = (payload.get("pin") or "").strip()
    if track_id is None or person_id is None or not pin:
        return {"ok": False, "error": "track_id, person_id, and pin are required"}

    person = ui.db.get_person(person_id)
    if not person:
        return {"ok": False, "granted": False, "error": "No such person"}
    if not ui.db.person_has_pin(person_id):
        return {"ok": True, "granted": False, "error": "No PIN set for this person"}
    if not ui.db.verify_person_pin(person_id, pin):
        name = f"{person['first_name']} {person['last_name']}".strip()
        ui._log_event("info", "PIN entry failed", name)
        # They resemble someone enrolled (that's the only way a PIN
        # prompt exists at all) but typed the wrong code — always ask
        # the emergency contacts what to do, regardless of the
        # unknown_text/unknown_call toggles above.
        ui.notify_pin_failure(name)
        return {"ok": True, "granted": False, "error": "Incorrect PIN"}

    if not ui._engine:
        return {"ok": False, "error": "Engine not attached — can't confirm live"}

    granted = ui._engine.confirm_via_pin(int(track_id), int(person_id))
    if granted:
        ui._log_event("identity", "Identity confirmed via PIN",
                      f"{person['first_name']} {person['last_name']}")
    return {"ok": True, "granted": granted}


# ── Voice enrollment (live mic, one sample per request) ────────────────────
#
# Flow from the UI: call /voice/sample repeatedly (each call blocks for
# ~4s while it records), then /voice/finish to save everything captured
# so far, or /voice/cancel to throw it away.

@app.post("/api/people/{person_id}/voice/sample")
async def voice_sample(person_id: int):
    ui, err = _require_db()
    if err:
        return err
    if not _VOICE_LAYER_AVAILABLE:
        return {"ok": False, "error": f"voice_layer unavailable: "
                                       f"{_VOICE_LAYER_IMPORT_ERROR}"}
    if not ui.db.get_person(person_id):
        return {"ok": False, "error": "No such person"}

    def _record():
        audio = record_audio()
        return embed_audio(audio)

    try:
        emb = await run_in_threadpool(_record)
    except Exception as e:
        return {"ok": False, "error": f"Recording failed: {e}"}

    if emb is None:
        with ui._voice_buffer_lock:
            count = len(ui._voice_buffer.get(person_id, []))
        return {"ok": False, "error": "Too quiet / no speech detected — "
                                       "try again, speaking normally.",
                "count": count}

    with ui._voice_buffer_lock:
        ui._voice_buffer.setdefault(person_id, []).append(emb)
        count = len(ui._voice_buffer[person_id])
    return {"ok": True, "count": count}


@app.get("/api/people/{person_id}/voice/status")
async def voice_status(person_id: int):
    ui, err = _require_db()
    if err:
        return err
    with ui._voice_buffer_lock:
        count = len(ui._voice_buffer.get(person_id, []))
    return {"ok": True, "count": count}


@app.post("/api/people/{person_id}/voice/cancel")
async def voice_cancel(person_id: int):
    ui, err = _require_db()
    if err:
        return err
    with ui._voice_buffer_lock:
        ui._voice_buffer.pop(person_id, None)
    return {"ok": True}


@app.post("/api/people/{person_id}/voice/finish")
async def voice_finish(person_id: int):
    ui, err = _require_db()
    if err:
        return err
    person = ui.db.get_person(person_id)
    if not person:
        return {"ok": False, "error": "No such person"}

    with ui._voice_buffer_lock:
        embeddings = ui._voice_buffer.pop(person_id, [])
    if not embeddings:
        return {"ok": False, "error": "No samples recorded yet — "
                                       "call /voice/sample first."}

    os.makedirs(VOICE_DIR, exist_ok=True)
    safe_name = person["first_name"].lower()
    save_path = os.path.join(VOICE_DIR, f"{safe_name}_{person_id}.npy")

    def _save():
        arr = np.stack(embeddings)
        np.save(save_path, arr)
        ui.db.save_voiceprint_path(person_id, save_path)

    await run_in_threadpool(_save)
    ui._reload_engine()
    ui._log_event("voice", "Voice enrolled",
                  f"{person['first_name']} {person['last_name']} "
                  f"— {len(embeddings)} sample(s)")
    return {"ok": True, "count": len(embeddings), "path": save_path}


# ── Face enrollment ──────────────────────────────────────────────────────
#
# Two ways in: capture straight from the live camera feed (needs
# ui.push_frame(...) to be wired up in your camera loop), or upload a
# photo directly from the browser.

@app.post("/api/people/{person_id}/face/capture")
async def face_capture(person_id: int):
    ui, err = _require_db()
    if err:
        return err
    if not _FACE_LAYER_AVAILABLE:
        return {"ok": False, "error": f"face_layer unavailable: "
                                       f"{_FACE_LAYER_IMPORT_ERROR}"}
    person = ui.db.get_person(person_id)
    if not person:
        return {"ok": False, "error": "No such person"}

    jpeg = ui.get_latest_jpeg()
    if jpeg is None:
        return {"ok": False, "error": "No live camera frame available — "
                                       "is main.py running with the "
                                       "camera loop calling ui.push_frame()?"}

    def _capture():
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        found = encode_face_from_frame(frame, jitter=ENCODING_JITTER_ENROLL)
        return [enc for _, enc in found]

    encodings = await run_in_threadpool(_capture)
    if not encodings:
        return {"ok": False, "error": "No face detected in the current "
                                       "camera frame — face the camera "
                                       "and try again."}
    if len(encodings) > 1:
        return {"ok": False, "error": "More than one face in frame — "
                                       "make sure only this person is "
                                       "visible, then try again."}

    total = ui.db.add_face_encodings(person_id, encodings)
    ui._reload_engine()
    ui._log_event("face", "Face sample captured",
                  f"{person['first_name']} {person['last_name']} "
                  f"— {total} total sample(s)")
    return {"ok": True, "count": total}


@app.post("/api/people/{person_id}/face/upload")
async def face_upload(person_id: int, file: UploadFile = File(...)):
    ui, err = _require_db()
    if err:
        return err
    if not _FACE_LAYER_AVAILABLE:
        return {"ok": False, "error": f"face_layer unavailable: "
                                       f"{_FACE_LAYER_IMPORT_ERROR}"}
    person = ui.db.get_person(person_id)
    if not person:
        return {"ok": False, "error": "No such person"}

    raw = await file.read()

    def _process():
        # save alongside known_faces/<first_name>/ too, so this stays
        # compatible with the folder-based enroll_person_faces() flow
        folder = os.path.join("known_faces", person["first_name"].lower())
        os.makedirs(folder, exist_ok=True)
        ext = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
        dest = os.path.join(folder, f"web_{int(time.time())}{ext}")
        with open(dest, "wb") as f:
            f.write(raw)
        found = encode_face_from_image(dest, jitter=ENCODING_JITTER_ENROLL)
        return found

    encodings = await run_in_threadpool(_process)
    if not encodings:
        return {"ok": False, "error": "No face found in that photo."}

    total = ui.db.add_face_encodings(person_id, encodings)
    ui._reload_engine()
    ui._log_event("face", "Face photo uploaded",
                  f"{person['first_name']} {person['last_name']} "
                  f"— {total} total sample(s)")
    return {"ok": True, "count": total}


@app.post("/api/people/{person_id}/face/enroll-path")
async def face_enroll_path(person_id: int, payload: dict):
    """Body: {"path": "known_faces/joseph", "append": false}.
    Bulk-trains face data from every photo/video already in that
    folder — the same thing enroll_person.py does from the terminal,
    just triggered from the web UI."""
    ui, err = _require_db()
    if err:
        return err
    if not _FACE_LAYER_AVAILABLE:
        return {"ok": False, "error": f"face_layer unavailable: "
                                       f"{_FACE_LAYER_IMPORT_ERROR}"}
    person = ui.db.get_person(person_id)
    if not person:
        return {"ok": False, "error": "No such person"}

    path = (payload.get("path") or "").strip()
    if not path:
        return {"ok": False, "error": "No folder path given"}
    append = bool(payload.get("append", False))

    def _train():
        return enroll_person_faces_from_folder(person_id, ui.db, path,
                                                append=append)

    n = await run_in_threadpool(_train)
    if n == 0:
        return {"ok": False, "error": f"No usable face photos/videos "
                                       f"found in '{path}'."}
    ui._reload_engine()
    ui._log_event("face", "Face trained from folder",
                  f"{person['first_name']} {person['last_name']} "
                  f"\u2014 {path} ({n} sample(s))")
    return {"ok": True, "count": n}


@app.post("/api/people/{person_id}/voice/enroll-path")
async def voice_enroll_path(person_id: int, payload: dict):
    """Body: {"path": "known_faces/joseph", "append": false}.
    Bulk-embeds every audio file already in that folder."""
    ui, err = _require_db()
    if err:
        return err
    if not _VOICE_LAYER_AVAILABLE:
        return {"ok": False, "error": f"voice_layer unavailable: "
                                       f"{_VOICE_LAYER_IMPORT_ERROR}"}
    person = ui.db.get_person(person_id)
    if not person:
        return {"ok": False, "error": "No such person"}

    path = (payload.get("path") or "").strip()
    if not path:
        return {"ok": False, "error": "No folder path given"}
    append = bool(payload.get("append", False))

    def _train():
        return enroll_person_voice_from_folder(person_id, ui.db, path,
                                                append=append)

    n = await run_in_threadpool(_train)
    if n == 0:
        return {"ok": False, "error": f"No usable audio files found "
                                       f"in '{path}'."}
    ui._reload_engine()
    ui._log_event("voice", "Voice trained from folder",
                  f"{person['first_name']} {person['last_name']} "
                  f"\u2014 {path} ({n} sample(s))")
    return {"ok": True, "count": n}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    ui: Optional[NovaWebUI] = getattr(app.state, "ui", None)
    try:
        if ui:
            await ws.send_json({"type": "snapshot", "data": ui.get_snapshot()})
        while True:
            # We don't expect inbound messages yet, but keep the socket
            # alive and ignore anything the client sends.
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ── Standalone demo (no camera / mic / db required) ───────────────────────────

def _demo(ui: NovaWebUI):
    time.sleep(1.5)
    ui.set_status("Motion detected...", OrbState.DETECTED)
    ui._log_event("face", "Face detected", "Track 01 — tracking")
    time.sleep(2)

    ui.set_status("Identifying...", OrbState.IDENTIFYING)
    time.sleep(2.5)

    ui.push_verdict(Verdict(
        track_id=1, name="Joseph Wella", person_id=1,
        face_conf=0.98, voice_conf=0.94, combined_conf=0.992,
        method="face + voice", access="full", decision="granted",
        top_desc="white Madrid jersey", bottom_desc="grey shorts",
        gender="male",
    ))
    ui.set_status("Joseph confirmed — access granted", OrbState.CONFIRMED)
    time.sleep(5)

    ui.trigger_wake_word()
    time.sleep(3)

    ui.set_status("Nova speaking...", OrbState.SPEAKING)
    time.sleep(3.5)
    ui.set_status("Monitoring...", OrbState.IDLE)
    time.sleep(2)

    ui.push_verdict(Verdict(
        track_id=2, name="Unknown", person_id=None,
        face_conf=0.42, voice_conf=0.0, combined_conf=0.42,
        method="face", access="none", decision="denied",
        top_desc="blue shirt", bottom_desc="dark jeans",
        gender="male", needs_voice=True,
    ))
    ui.set_status("Unknown person — voice verification required",
                  OrbState.WAITING)
    time.sleep(4)

    ui.set_status("Voice did not match — access denied", OrbState.DENIED)
    time.sleep(2.5)
    ui.remove_track(2)
    ui.set_status("Monitoring...", OrbState.IDLE)

    # loop the demo forever so the UI never looks "dead"
    while True:
        time.sleep(6)
        ui.trigger_wake_word()
        time.sleep(6)
        ui.set_status("Monitoring...", OrbState.IDLE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                         help="Run with simulated events (no camera/mic/db)")
    parser.add_argument("--mic", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    ui = NovaWebUI(mic_device=args.mic)

    if args.demo:
        threading.Thread(target=_demo, args=(ui,), daemon=True).start()

    ui.run(port=args.port)

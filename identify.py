import cv2
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from db import NovaDB
from face_layer import FaceLayer
from voice_layer import VoiceLayer
from clothing_layer import ClothingLayer
from web.server import Verdict, OrbState, ALARM_TRIGGER_SECONDS

WEIGHT_FACE = 0.50
WEIGHT_VOICE = 0.35
WEIGHT_CLOTHING = 0.15
ACCESS_THRESHOLD = 0.50
FACE_NAME_THRESHOLD = ACCESS_THRESHOLD
TRACK_TIMEOUT = 8.0
CLOTHING_SAMPLE_INTERVAL = 5
VOICE_PROMPT_THRESHOLD = 0.50
PRESENCE_COOLDOWN = 30 * 60
PIN_FACE_MIN = 0.40
PIN_VOICE_MIN = 0.65

@dataclass
class Track:
    track_id: int
    person_id: Optional[int] = None
    name: str = "Unknown"
    face_conf: float = 0.0
    voice_conf: float = 0.0
    clothing_conf: float = 0.0
    candidate_person_id: Optional[int] = None
    candidate_name: str = ""
    runner_up_name: str = ""
    ambiguous: bool = False
    pin_required: bool = False
    face_distance: float = 1.0
    beard_visible: bool = False
    gender: str = "unknown"
    top_desc: str = "not visible"
    bottom_desc: str = "not visible"
    method: str = ""
    conflict: bool = False
    needs_voice: bool = False
    composite: float = 0.0
    decision: str = "denied"
    access: str = "none"
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 0
    location: Optional[tuple] = None

    def to_verdict(self) -> Verdict:
        return Verdict(
            track_id=self.track_id,
            name=self.name,
            person_id=self.person_id,
            face_conf=self.face_conf,
            voice_conf=self.voice_conf,
            combined_conf=self.composite,
            method=self.method,
            access=self.access,
            decision=self.decision,
            top_desc=self.top_desc,
            bottom_desc=self.bottom_desc,
            gender=self.gender,
            conflict=self.conflict,
            needs_voice=self.needs_voice,
            pin_required=self.pin_required,
            candidate_person_id=self.candidate_person_id,
            candidate_name=self.candidate_name,
            runner_up_name=self.runner_up_name,
            ambiguous=self.ambiguous,
        )

class IdentifyEngine:
    def __init__(self, db: NovaDB,
                 ui=None,
                 brain=None,
                 voice_device: int = 1):
        self.db = db
        self.ui = ui
        self.brain = brain
        self._lock = threading.Lock()
        self._face = FaceLayer(db)
        self._voice = VoiceLayer(db, on_voice_result=self._on_voice_result)
        self._clothing = ClothingLayer(db)
        self._tracks: dict[int, Track] = {}
        self._next_tid = 1
        self._frame_count = 0
        self._voice_prompted: dict[int, float] = {}
        self._alerted_tracks: set = set()
        self._unknown_only_since: Optional[float] = None
        self._presence: dict[int, dict] = {}

    def start(self):
        self._voice.start()
        print("[IdentifyEngine] Running.")

    def stop(self):
        self._voice.stop()

    def reload_people(self):
        self._face.reload()
        self._voice.reload()

    def confirm_via_pin(self, track_id: int, person_id: int) -> bool:
        with self._lock:
            track = self._tracks.get(track_id)
            if not track or track.candidate_person_id != person_id:
                return False
            p = self.db.get_person(person_id)
            if not p:
                return False
            now = time.time()
            track.person_id = person_id
            track.name = f"{p['first_name']} {p['last_name']}".strip()
            track.access = p.get("access_level", "none")
            track.decision = "granted" if track.access in ("full", "limited") else "denied"
            track.pin_required = False
            track.method = (_method_string(track) + " + pin").strip(" +")
            if track.decision == "granted":
                self._presence[person_id] = {"name": track.name, "confirmed_at": now}
            return track.decision == "granted"

    def _on_voice_result(self, person_id, name, similarity, is_known):
        if self.ui and hasattr(self.ui, "push_voice_status"):
            self.ui.push_voice_status(name, similarity, is_known)

        with self._lock:
            attached = False
            for track in self._tracks.values():
                if is_known:
                    if (track.person_id == person_id or
                            track.name == "Unknown"):
                        track.voice_conf = similarity
                        if track.person_id is None:
                            p = self.db.get_person(person_id)
                            if p:
                                track.person_id = person_id
                                track.name = (f"{p['first_name']} "
                                              f"{p['last_name']}".strip())
                        attached = True
                        break
                else:
                    if track.name == "Unknown":
                        track.voice_conf = 0.0

            if (is_known and not attached and
                    not any(t.person_id == person_id for t in self._tracks.values())):
                p = self.db.get_person(person_id)
                if p:
                    tid = self._next_tid
                    self._next_tid += 1
                    track = Track(track_id=tid)
                    track.person_id = person_id
                    track.name = f"{p['first_name']} {p['last_name']}".strip()
                    track.voice_conf = similarity
                    track.method = "voice"
                    track.composite = _composite(0.0, similarity, 0.0)
                    track.access = p.get("access_level", "none")
                    track.decision = ("granted"
                                      if track.composite >= ACCESS_THRESHOLD
                                      and track.access in ("full", "limited")
                                      else "denied")
                    track.last_seen = time.time()
                    self._tracks[tid] = track
                    if track.decision == "granted":
                        self._presence[person_id] = {
                            "name": track.name, "confirmed_at": time.time()}
                    if self.ui:
                        self.ui.push_verdict(track.to_verdict())
                        self.ui._log_event(
                            "voice", "Voice-only identification",
                            f"{track.name} \u2014 no face track available")

    def process_frame(self, frame_bgr: np.ndarray) -> list[Verdict]:
        self._frame_count += 1
        now = time.time()

        face_results = self._face.identify(frame_bgr)

        with self._lock:
            matched_ids = set()

            for face_result in face_results:
                loc = face_result["location"]
                tid = self._match_or_create_track(loc)
                track = self._tracks[tid]
                matched_ids.add(tid)

                track.last_seen = now
                track.frame_count += 1
                track.location = loc

                track.runner_up_name = face_result.get("runner_up_name") or ""
                track.ambiguous = bool(face_result.get("ambiguous"))

                if face_result["is_known"]:
                    track.candidate_person_id = face_result["person_id"]
                    track.candidate_name = face_result["name"]
                else:
                    track.candidate_person_id = None
                    track.candidate_name = ""

                presence = (self._presence.get(track.candidate_person_id)
                            if track.candidate_person_id else None)
                presence_valid = (presence is not None and
                                  now - presence["confirmed_at"] < PRESENCE_COOLDOWN)

                if presence_valid:
                    track.face_conf = max(face_result.get("confidence", 0.0),
                                          track.face_conf)
                    track.person_id = track.candidate_person_id
                    track.name = presence["name"]
                elif face_result["is_known"] and \
                        face_result["confidence"] >= FACE_NAME_THRESHOLD:
                    track.face_conf = face_result["confidence"]
                    track.person_id = face_result["person_id"]
                    track.name = face_result["name"]
                elif face_result["is_known"]:
                    track.face_conf = face_result["confidence"]
                    if track.voice_conf < 0.50:
                        track.name = "Unknown"
                        track.person_id = None
                else:
                    track.face_conf = 0.0
                    if track.voice_conf < 0.50:
                        track.name = "Unknown"
                        track.person_id = None

                if self._frame_count % CLOTHING_SAMPLE_INTERVAL == 0:
                    x, y, w, h = self._face.location_to_box(loc)
                    zones = self._clothing.process(frame_bgr, (x, y, w, h))
                    track.top_desc = _zone_colour(zones, "torso")
                    track.bottom_desc = _zone_colour(zones, "legs")

                    if track.person_id and track.face_conf >= 0.70:
                        self._clothing.save_to_wardrobe(
                            track.person_id, zones,
                            label=f"auto_{track.name}"
                        )
                        track.clothing_conf = track.face_conf
                    elif track.name == "Unknown":
                        pid, cname, cscore = self._clothing.identify(zones)
                        if pid and cscore >= 0.65:
                            p = self.db.get_person(pid)
                            if p:
                                track.person_id = pid
                                track.name = (f"{p['first_name']} "
                                              f"{p['last_name']}".strip())
                                track.clothing_conf = cscore
                                track.access = p.get("access_level", "none")

                self._voice.set_face_confidence(track.face_conf)

                last_prompt = self._voice_prompted.get(tid, 0.0)
                if (track.face_conf < VOICE_PROMPT_THRESHOLD and
                        track.voice_conf < 0.50 and
                        now - last_prompt > 20.0):
                    track.needs_voice = True
                    self._voice_prompted[tid] = now
                    if self.brain:
                        self.brain.trigger_voice_prompt()
                else:
                    track.needs_voice = False

                track.composite = _composite(
                    track.face_conf,
                    track.voice_conf,
                    track.clothing_conf,
                )

                if track.name != "Unknown" and track.composite >= ACCESS_THRESHOLD:
                    p = self.db.get_person_by_name(track.name.split()[0])
                    track.access = p["access_level"] if p else "none"
                    track.decision = "granted" if track.access in ("full", "limited") else "denied"
                    track.method = _method_string(track)
                    track.pin_required = False
                    if track.decision == "granted" and track.person_id:
                        self._presence[track.person_id] = {
                            "name": track.name, "confirmed_at": now,
                        }
                else:
                    track.decision = "denied"
                    track.access = "none"
                    track.method = "face" if track.face_conf > 0 else "unknown"
                    track.pin_required = bool(
                        track.candidate_person_id and
                        (track.face_conf > PIN_FACE_MIN or
                         track.voice_conf >= PIN_VOICE_MIN)
                    )

            stale = [tid for tid, t in self._tracks.items()
                     if now - t.last_seen > TRACK_TIMEOUT
                     and tid not in matched_ids]
            for tid in stale:
                if self.ui:
                    self.ui.remove_track(tid)
                self._alerted_tracks.discard(tid)
                del self._tracks[tid]

            _enforce_singularity(list(self._tracks.values()))

            for tid, track in self._tracks.items():
                if track.name != "Unknown":
                    continue
                desc = (f"{track.gender}, {track.top_desc}, "
                        f"{track.bottom_desc}")
                if tid not in self._alerted_tracks and track.frame_count > 10:
                    self._alerted_tracks.add(tid)
                    if self.brain:
                        self.brain.trigger_unknown_alert(tid, desc)
                    if self.ui and hasattr(self.ui, "notify_unknown_detected"):
                        self.ui.notify_unknown_detected(tid, desc)
                if (tid in self._alerted_tracks and self.ui and
                        hasattr(self.ui, "notify_unknown_persists")):
                    self.ui.notify_unknown_persists(tid, desc)

            has_known = any(t.name != "Unknown" for t in self._tracks.values())
            has_unknown = any(t.name == "Unknown" for t in self._tracks.values())
            alarm_active = False
            if has_unknown and not has_known:
                if self._unknown_only_since is None:
                    self._unknown_only_since = now
                elapsed = now - self._unknown_only_since
                if elapsed >= ALARM_TRIGGER_SECONDS:
                    if self.ui and hasattr(self.ui, "trigger_alarm"):
                        self.ui.trigger_alarm()
                    alarm_active = True
            else:
                self._unknown_only_since = None
                if self.ui and hasattr(self.ui, "clear_alarm"):
                    self.ui.clear_alarm()
            if not alarm_active and self.ui and getattr(self.ui, "_alarm_active", False):
                alarm_active = True

            if self.brain:
                self.brain.update_observation(
                    _build_observation(list(self._tracks.values())))

            verdicts = []
            for track in self._tracks.values():
                v = track.to_verdict()
                verdicts.append(v)
                if self.ui:
                    self.ui.push_verdict(v)

            if self.ui:
                if alarm_active:
                    pass
                elif any(t.conflict for t in self._tracks.values()):
                    self.ui.set_status("Identity conflict detected",
                                       OrbState.DENIED)
                elif any(t.pin_required for t in self._tracks.values()):
                    self.ui.set_status("Not quite sure — enter your PIN to confirm",
                                       OrbState.PIN_REQUIRED)
                elif any(t.name == "Unknown" and not t.pin_required
                         for t in self._tracks.values()):
                    self.ui.set_status("Unknown person in frame",
                                       OrbState.ALERT if hasattr(OrbState, 'ALERT')
                                       else OrbState.DENIED)
                elif any(t.needs_voice for t in self._tracks.values()):
                    self.ui.set_status("Waiting for voice confirmation",
                                       OrbState.WAITING)
                elif any(t.decision == "granted" for t in self._tracks.values()):
                    self.ui.set_status("Identity confirmed",
                                       OrbState.CONFIRMED)
                elif self._tracks:
                    self.ui.set_status("Identifying...",
                                       OrbState.IDENTIFYING)
                else:
                    self.ui.set_status("Monitoring...", OrbState.IDLE)

            return verdicts

    def _match_or_create_track(self, location: tuple) -> int:
        top, right, bottom, left = location
        cx = (left + right) / 2
        cy = (top + bottom) / 2

        best_tid = None
        best_dist = float("inf")

        for tid, track in self._tracks.items():
            if track.location is None:
                continue
            pt, pr, pb, pl = track.location
            tx = (pl + pr) / 2
            ty = (pt + pb) / 2
            dist = ((cx - tx)**2 + (cy - ty)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is not None and best_dist < 120:
            return best_tid

        tid = self._next_tid
        self._next_tid += 1
        self._tracks[tid] = Track(track_id=tid)

        if self.ui:
            self.ui.set_status("New person detected", OrbState.DETECTED)

        return tid

def _composite(face: float, voice: float, clothing: float) -> float:
    active = []
    if face > 0:
        active.append(("face", face, WEIGHT_FACE))
    if voice > 0:
        active.append(("voice", voice, WEIGHT_VOICE))
    if clothing > 0:
        active.append(("clothing", clothing, WEIGHT_CLOTHING))

    if not active:
        return 0.0

    total_w = sum(w for _, _, w in active)
    score = sum(v * w for _, v, w in active) / total_w
    return round(min(score, 1.0), 3)

def _enforce_singularity(tracks: list[Track]):
    seen: dict[str, Track] = {}
    for track in tracks:
        if track.name in ("Unknown", ""):
            continue
        if track.name in seen:
            existing = seen[track.name]
            if track.composite > existing.composite:
                existing.name = "Unknown"
                existing.conflict = True
                existing.decision = "denied"
                existing.person_id = None
                seen[track.name] = track
            else:
                track.name = "Unknown"
                track.conflict = True
                track.decision = "denied"
                track.person_id = None
        else:
            seen[track.name] = track

def _method_string(track: Track) -> str:
    parts = []
    if track.face_conf > 0:
        parts.append("face")
    if track.voice_conf > 0:
        parts.append("voice")
    if track.clothing_conf > 0:
        parts.append("clothing")
    return " + ".join(parts) if parts else "unknown"

def _zone_colour(zones: dict, zone_name: str) -> str:
    from clothing_layer import _dominant_colour
    hist = zones.get(zone_name)
    if hist is None:
        return "not visible"
    return _dominant_colour(hist) + " " + {
        "torso": "top", "legs": "bottom", "shoes": "shoes"
    }.get(zone_name, zone_name)

def _build_observation(tracks: list[Track]) -> str:
    if not tracks:
        return "[OBSERVATION]\nNo people detected. Room appears empty."
    lines = ["[OBSERVATION]"]
    for t in tracks:
        conflict_tag = " CONFLICT" if t.conflict else ""
        lines.append(f"Track {t.track_id:02d}: {t.name}{conflict_tag}")
        lines.append(f"  Composite : {t.composite:.0%}  Decision: {t.decision}")
        lines.append(f"  Face      : {t.face_conf:.0%}  Voice: {t.voice_conf:.0%}")
        lines.append(f"  Clothing  : {t.top_desc}, {t.bottom_desc}")
        lines.append(f"  Method    : {t.method}")
        lines.append("")
    return "\n".join(lines)

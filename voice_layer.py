import os
import threading
import time
import numpy as np
import sounddevice as sd
from audio_devices import configure_audio_device
configure_audio_device()

from resemblyzer import VoiceEncoder, preprocess_wav
from db import NovaDB

SAMPLE_RATE = 44100
RECORD_SECONDS = 4
ENROLL_SAMPLES = 15
SIMILARITY_THRESHOLD = 0.80
VOICE_DIR = "data/voiceprints"

INTERVAL_MID_CONF = 5.0
INTERVAL_LOW_CONF = 0.0
INTERVAL_NO_FACE = 3.0

FACE_HIGH = 0.80
FACE_LOW = 0.50

_encoder = None

def get_encoder() -> VoiceEncoder:
    global _encoder
    if _encoder is None:
        print("[VoiceLayer] Loading Resemblyzer encoder...")
        _encoder = VoiceEncoder()
        print("[VoiceLayer] Encoder ready.")
    return _encoder

def record_audio(seconds: float = RECORD_SECONDS,
                 samplerate: int = SAMPLE_RATE,
                 timeout_margin: float = 5.0) -> np.ndarray:
    n_frames = int(seconds * samplerate)
    buffer = np.zeros(n_frames, dtype="float32")
    frames_written = 0
    done = threading.Event()

    def callback(indata, frames, time_info, status):
        nonlocal frames_written
        remaining = n_frames - frames_written
        take = min(remaining, frames)
        if take > 0:
            buffer[frames_written:frames_written + take] = indata[:take, 0]
            frames_written += take
        if frames_written >= n_frames:
            done.set()
            raise sd.CallbackStop()

    try:
        stream = sd.InputStream(samplerate=samplerate, channels=1,
                                dtype="float32", callback=callback)
        with stream:
            finished = done.wait(timeout=seconds + timeout_margin)
    except sd.PortAudioError as e:
        raise RuntimeError(
            f"Could not open the microphone ({e}). Run "
            f"'python3 audio_devices.py' to see available devices, then "
            f"set NOVA_INPUT_DEVICE=<index> if the wrong one was picked."
        ) from e

    if not finished:
        got = frames_written / samplerate
        raise RuntimeError(
            f"Recording timed out after {seconds + timeout_margin:.0f}s — "
            f"the input stream opened but only delivered {got:.1f}s of "
            f"{seconds:.0f}s of audio. This almost always means the wrong "
            f"device is selected (a stream that 'opens' but never really "
            f"produces data), or something else is holding the mic. Run "
            f"'python3 audio_devices.py --test-all' to see which device "
            f"index actually picks up sound, then set "
            f"NOVA_INPUT_DEVICE=<that index>."
        )

    return buffer

def is_speech(audio: np.ndarray, threshold: float = 0.008) -> bool:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms > threshold

def embed_audio(audio: np.ndarray,
                samplerate: int = SAMPLE_RATE):
    if not is_speech(audio):
        return None
    try:
        wav = preprocess_wav(audio, source_sr=samplerate)
        if len(wav) < samplerate * 0.5:
            return None
        return get_encoder().embed_utterance(wav)
    except Exception as e:
        print(f"[VoiceLayer][WARN] Embedding failed: {e}")
        return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def enroll_person_voice(person_id: int, db: NovaDB,
                        n_samples: int = ENROLL_SAMPLES) -> bool:
    person = db.get_person(person_id)
    if not person:
        print(f"[VoiceLayer][ERROR] No person with id={person_id}")
        return False

    name = f"{person['first_name']} {person['last_name']}".strip()
    safe_name = person["first_name"].lower()

    print(f"\n[VOICE ENROLL] {name}")
    print(f"  You will record {n_samples} samples of {RECORD_SECONDS}s each.")
    print("  Speak naturally — vary your sentences across samples.")
    print("  This captures the natural range of your voice.\n")

    os.makedirs(VOICE_DIR, exist_ok=True)
    embeddings = []
    i = 1
    while i <= n_samples:
        input(f"  Press Enter for sample {i}/{n_samples}, then speak...")
        print(f"  Recording {RECORD_SECONDS}s — speak now...")
        audio = record_audio()
        emb = embed_audio(audio)
        if emb is None:
            print("  [WARN] Could not embed — too quiet? Retrying.")
            continue
        embeddings.append(emb)
        print(f"  [OK] Sample {i} captured.\n")
        i += 1

    embeddings_array = np.stack(embeddings)
    save_path = os.path.join(VOICE_DIR, f"{safe_name}_{person_id}.npy")
    np.save(save_path, embeddings_array)
    db.save_voiceprint_path(person_id, save_path)
    print(f"[VOICE ENROLL] Done. {len(embeddings)} embeddings saved -> {save_path}")
    return True

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

def enroll_person_voice_from_folder(person_id: int, db: NovaDB,
                                    folder: str,
                                    append: bool = False) -> int:
    person = db.get_person(person_id)
    if not person:
        print(f"[VoiceLayer][ERROR] No person with id={person_id}")
        return 0

    if not os.path.isdir(folder):
        print(f"[VoiceLayer][ERROR] Folder not found: {folder}")
        return 0

    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS
    )
    if not files:
        print(f"[VoiceLayer] No audio files found in {folder} "
              f"(looked for {', '.join(sorted(AUDIO_EXTS))})")
        return 0

    name = f"{person['first_name']} {person['last_name']}".strip()
    print(f"\n[VOICE ENROLL — FOLDER] {name}")
    print(f"  {len(files)} audio file(s) found in {folder}")

    embeddings = []
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            wav = preprocess_wav(path)
            if len(wav) < 8000:
                print(f"  [WARN] {fname} too short — skipped")
                continue
            emb = get_encoder().embed_utterance(wav)
            embeddings.append(emb)
            print(f"  [OK] {fname}")
        except Exception as e:
            print(f"  [WARN] {fname} — could not embed: {e}")

    if not embeddings:
        print(f"  [ERROR] No usable audio found in {folder}.")
        return 0

    if append:
        existing = db.load_voiceprint(person_id)
        if existing is not None:
            if existing.ndim == 1:
                existing = existing[np.newaxis, :]
            print(f"  [INFO] Appending to {len(existing)} existing sample(s)")
            embeddings = list(existing) + embeddings

    safe_name = person["first_name"].lower()
    os.makedirs(VOICE_DIR, exist_ok=True)
    save_path = os.path.join(VOICE_DIR, f"{safe_name}_{person_id}.npy")
    np.save(save_path, np.stack(embeddings))
    db.save_voiceprint_path(person_id, save_path)

    print(f"[VOICE ENROLL — FOLDER] Done. {len(embeddings)} total "
          f"embedding(s) saved -> {save_path}")
    return len(embeddings)

class VoiceResult:
    def __init__(self):
        self._lock = threading.Lock()
        self._person_id = None
        self._name = "Unknown"
        self._similarity = 0.0
        self._is_known = False
        self._timestamp = 0.0
        self._fresh = False

    def update(self, person_id, name, similarity, is_known):
        with self._lock:
            self._person_id = person_id
            self._name = name
            self._similarity = similarity
            self._is_known = is_known
            self._timestamp = time.time()
            self._fresh = True

    def read(self) -> dict:
        with self._lock:
            self._fresh = False
            return {
                "person_id": self._person_id,
                "name": self._name,
                "similarity": self._similarity,
                "is_known": self._is_known,
                "timestamp": self._timestamp,
                "age_seconds": time.time() - self._timestamp,
            }

    def is_fresh(self) -> bool:
        with self._lock:
            return self._fresh

    def age(self) -> float:
        with self._lock:
            return time.time() - self._timestamp

class VoiceLayer:
    def __init__(self, db: NovaDB,
                 threshold: float = SIMILARITY_THRESHOLD,
                 on_voice_result=None):
        self.db = db
        self.threshold = threshold
        self._on_voice_result_cb = on_voice_result
        self.result = VoiceResult()
        self._known = []
        self._face_confidence = 0.0
        self._face_lock = threading.Lock()
        self._running = False
        self._thread = None
        self.reload()

    def reload(self):
        self._known = self.db.load_all_voiceprints()
        total = sum(
            vp.shape[0] if vp.ndim == 2 else 1
            for _, _, vp in self._known
        )
        print(f"[VoiceLayer] Loaded {len(self._known)} person(s), "
              f"{total} total embedding(s).")

        if self._known and not self._running:
            self.start()

    def set_face_confidence(self, confidence: float):
        with self._face_lock:
            self._face_confidence = confidence

    def _get_face_confidence(self) -> float:
        with self._face_lock:
            return self._face_confidence

    def _decide_interval(self):
        conf = self._get_face_confidence()
        if conf >= FACE_HIGH:
            return None
        elif conf >= FACE_LOW:
            return INTERVAL_MID_CONF
        elif conf > 0:
            return INTERVAL_LOW_CONF
        else:
            return INTERVAL_NO_FACE

    def _identify_from_audio(self, audio: np.ndarray):
        emb = embed_audio(audio)
        if emb is None:
            return

        best_id = None
        best_name = "Unknown"
        best_sim = 0.0

        for pid, name, vp in self._known:
            if vp.ndim == 1:
                vp = vp[np.newaxis, :]

            for stored_emb in vp:
                sim = cosine_similarity(emb, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid
                    best_name = name

        is_known = best_sim >= self.threshold
        if not is_known:
            best_id = None
            best_name = "Unknown"

        self.result.update(best_id, best_name, best_sim, is_known)
        status = f"✓ {best_name}" if is_known else "✗ Unknown"
        print(f"[VoiceLayer] {status}  similarity={best_sim:.3f}")

        if self._on_voice_result_cb:
            try:
                self._on_voice_result_cb(best_id, best_name,
                                         best_sim, is_known)
            except Exception as e:
                print(f"[VoiceLayer] Callback error: {e}")

    def _run(self):
        print("[VoiceLayer] Background thread started.")
        last_run = 0.0

        while self._running:
            interval = self._decide_interval()

            if interval is None:
                time.sleep(1.0)
                continue

            now = time.time()
            if now - last_run < interval:
                time.sleep(0.1)
                continue

            try:
                audio = record_audio()
                self._identify_from_audio(audio)
                last_run = time.time()
            except Exception as e:
                print(f"[VoiceLayer][ERROR] {e}")
                time.sleep(10.0)

    def start(self):
        if not self._known:
            print("[VoiceLayer][WARN] No voiceprints loaded — voice layer inactive.")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[VoiceLayer] Running.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[VoiceLayer] Stopped.")

if __name__ == "__main__":
    import sys

    db = NovaDB()

    if len(sys.argv) > 1 and sys.argv[1] == "enroll":
        enroll_person_voice(int(sys.argv[2]), db)

    elif len(sys.argv) > 1 and sys.argv[1] == "enroll-folder":
        if len(sys.argv) < 4:
            print("Usage: python3 voice_layer.py enroll-folder "
                  "<person_id> <folder> [--append]")
            sys.exit(1)
        pid = int(sys.argv[2])
        folder = sys.argv[3]
        append = "--append" in sys.argv[4:]
        n = enroll_person_voice_from_folder(pid, db, folder, append=append)
        sys.exit(0 if n > 0 else 1)

    elif len(sys.argv) > 1 and sys.argv[1] == "verify":
        known = db.load_all_voiceprints()
        if not known:
            print("[ERROR] No voiceprints in DB.")
            sys.exit(1)
        get_encoder()
        print("\n[TEST] Recording 4 seconds — speak now...")
        audio = record_audio()
        emb = embed_audio(audio)
        if emb is None:
            print("[ERROR] Could not embed audio.")
            sys.exit(1)
        print("\n── Results ──────────────────────────────────")
        for pid, name, vp in known:
            if vp.ndim == 1:
                vp = vp[np.newaxis, :]
            best = max(cosine_similarity(emb, e) for e in vp)
            tag = "✓ MATCH" if best >= SIMILARITY_THRESHOLD else "✗ no match"
            print(f"  {name:<20} best_similarity={best:.3f}  {tag}")
        print("─────────────────────────────────────────────")

    else:
        print("Usage:")
        print("  python3 voice_layer.py enroll <person_id>")
        print("      Live mic recording, one sample at a time.")
        print("  python3 voice_layer.py enroll-folder <person_id> <folder> [--append]")
        print("      Bulk-embed every audio file already in <folder>.")
        print("  python3 voice_layer.py verify")

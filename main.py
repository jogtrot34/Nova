import cv2
import os
import sys
import time
import threading
import argparse
import numpy as np

from db import NovaDB
from identify import IdentifyEngine
from web.server import NovaWebUI as NovaUI, OrbState

try:
    import sounddevice as sd
    from piper import PiperVoice
    _PIPER_AVAILABLE = True
except Exception:
    _PIPER_AVAILABLE = False

try:
    from brain import NovaBrain
    _BRAIN_AVAILABLE = True
except Exception:
    _BRAIN_AVAILABLE = False

PIPER_MODEL = "models/en_US-amy-medium.onnx"
PIPER_CONFIG = "models/en_US-amy-medium.onnx.json"
MIC_DEVICE = None
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

class Speaker:
    def __init__(self, model_path: str, config_path: str, ui=None):
        self._ui = ui
        self._queue = []
        self._lock = threading.Lock()
        self._voice = None
        self._ready = False
        threading.Thread(target=self._load,
                         args=(model_path, config_path),
                         daemon=True).start()

    def _load(self, model_path, config_path):
        try:
            self._voice = PiperVoice.load(model_path,
                                          config_path=config_path)
            self._ready = True
            print("[Speaker] Piper TTS ready.")
            threading.Thread(target=self._run, daemon=True).start()
        except Exception as e:
            print(f"[Speaker] Could not load Piper: {e}")

    def say(self, text: str):
        if not self._ready:
            print(f"[TTS] {text}")
            return
        with self._lock:
            self._queue.append(text)

    def _run(self):
        while True:
            text = None
            with self._lock:
                if self._queue:
                    text = self._queue.pop(0)
            if text:
                try:
                    if self._ui:
                        self._ui.set_speaking(True)
                    chunks = []
                    for chunk in self._voice.synthesize(text):
                        chunks.append(chunk.audio_float_array)
                    if chunks:
                        audio = np.concatenate(chunks)
                        sd.play(audio,
                                samplerate=self._voice.config.sample_rate)
                        sd.wait()
                except Exception as e:
                    print(f"[Speaker] Playback error: {e}")
                finally:
                    if self._ui:
                        self._ui.set_speaking(False)
            else:
                time.sleep(0.05)

class DummySpeaker:
    def say(self, text: str):
        print(f"[Nova says] {text}")

def terminal_input_thread(brain, speaker):
    print("[Chat] Type questions for Nova (terminal fallback).")
    while True:
        try:
            line = input()
            if line.strip():
                def on_response(text, sp=speaker):
                    print(f"\nNova: {text}\n")
                    sp.say(text)
                brain._on_response = on_response
                brain.ask(line.strip())
        except (EOFError, KeyboardInterrupt):
            break

def open_camera(source):
    import cv2

    try:
        idx = int(source)
    except ValueError:
        cap = cv2.VideoCapture(str(source))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            print(f"[Camera] Opened: {source}")
            return cap
        print(f"[Camera][WARN] Could not open {source!r}")
        return None

    cap = cv2.VideoCapture(idx)
    ok = cap.isOpened()
    if ok:
        ret, _ = cap.read()
        ok = ret

    if not ok:
        cap.release()
        print(f"[Camera][WARN] Camera index {idx} didn't deliver a "
              f"frame — auto-detecting a working camera...")
        from camera_devices import find_working_camera
        auto_idx = find_working_camera()
        if auto_idx is None:
            print("[Camera][ERROR] No working camera found on this "
                  "machine. Nova will keep running without live video "
                  "— check connections, close any other app using the "
                  "camera, and run 'python3 camera_devices.py' to test. "
                  "Everything else (PIN, emergency alerts, modem, web "
                  "UI) still works without it.")
            return None
        print(f"[Camera] Auto-detected a working camera at index {auto_idx}")
        cap = cv2.VideoCapture(auto_idx)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print(f"[Camera] Ready.")
    return cap

def main():
    parser = argparse.ArgumentParser(description="Nova Security AI")
    parser.add_argument("--camera", default="0",
                        help="Camera index or URL")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable Piper TTS output")
    parser.add_argument("--no-brain", action="store_true",
                        help="Disable LLM brain (faster startup)")
    parser.add_argument("--mic", type=int, default=MIC_DEVICE,
                        help="Microphone device index")
    args = parser.parse_args()

    print("=" * 55)
    print("  Nova -- AI Security Intelligence")
    print("  Mzuzu University, Dept. Physics & Electronics")
    print("=" * 55)

    db = NovaDB()
    db.summary()

    ui = NovaUI(mic_device=args.mic)

    if not args.no_voice and _PIPER_AVAILABLE:
        speaker = Speaker(PIPER_MODEL, PIPER_CONFIG, ui=ui)
    else:
        speaker = DummySpeaker()
        if args.no_voice:
            print("[Speaker] TTS disabled by --no-voice flag.")
        else:
            print("[Speaker] Piper not available, using terminal output.")

    brain = None
    if not args.no_brain and _BRAIN_AVAILABLE:
        def on_brain_response(text):
            print(f"\nNova: {text}")
            speaker.say(text)

        brain = NovaBrain(on_response=on_brain_response)
        print("[Brain] Qwen2.5-3B loading in background...")
    else:
        print("[Brain] LLM disabled.")

    engine = IdentifyEngine(db, ui=ui, brain=brain,
                            voice_device=args.mic)
    engine.start()

    ui.set_engine(engine)

    try:
        from audio_responses import VoiceBank
        ui.set_voice_bank(VoiceBank(ui=ui))
    except Exception as e:
        print(f"[Nova] VoiceBank not attached: {e}")

    def _try_connect_modem():
        try:
            from emergency_dial import EmergencyDialer
            from contacts_db import ContactsDB
            from notifier import Notifier
            dialer = EmergencyDialer()
            if dialer.connect():
                ui.set_notifier(Notifier(dialer, ContactsDB()))
                print("[Nova] Modem connected at startup.")
            else:
                print("[Nova] No modem found at startup — connect one "
                      "later from the Controls panel in the web UI.")
        except Exception as e:
            print(f"[Nova] Modem auto-connect skipped: {e}")

    threading.Thread(target=_try_connect_modem, daemon=True).start()

    if os.environ.get("NOVA_ENABLE_SAFE_WORD") == "1":
        try:
            from safe_word import SafeWordListener
            safe_word_listener = SafeWordListener(ui=ui)
            safe_word_listener.start()
        except Exception as e:
            print(f"[Nova] Safe word listener not started: {e}")
    else:
        print("[Nova] Safe word listener disabled (mic reserved for "
              "voice_layer.py verification). Set NOVA_ENABLE_SAFE_WORD=1 "
              "to turn it back on.")

    cap = open_camera(args.camera)

    if cap is None:
        ui._log_event("info", "No camera detected",
                      "Running without live video — run 'python3 "
                      "camera_devices.py' to test, or fix and restart. "
                      "Everything else still works.")
        print("[Camera] Skipping camera loop — no working camera found.")
    else:
        for _ in range(5):
            cap.read()

        def camera_loop():
            consecutive_failures = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures % 30 == 1:
                        print(f"[Camera] Frame read failed ({consecutive_failures}x).")
                    time.sleep(0.1)
                    continue
                consecutive_failures = 0
                try:
                    engine.process_frame(frame)
                    ui.push_frame(frame)
                except Exception as e:
                    print(f"[Camera] process_frame error: {e}")
                time.sleep(0.033)

        cam_thread = threading.Thread(target=camera_loop, daemon=True)
        cam_thread.start()
        print("[Camera] Loop started.")

    if brain:
        t = threading.Thread(
            target=terminal_input_thread,
            args=(brain, speaker),
            daemon=True)
        t.start()

    print("[Nova] Starting UI...")
    ui.run()

    cap.release()
    engine.stop()
    print("[Nova] Shutdown complete.")

if __name__ == "__main__":
    main()

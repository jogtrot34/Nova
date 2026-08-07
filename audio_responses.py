"""
audio_responses.py

Plays Nova's pre-recorded ElevenLabs voice lines instead of live TTS
for the phrases she says over and over. Record each line in ElevenLabs,
export as WAV, and drop it in assets/voice/ using the exact filenames
in the PHRASES dict below (full manifest + exact wording is in
README_VOICE_AND_CONTACTS.md).

File format: WAV, 16-bit PCM, mono, 44100Hz (ElevenLabs: export as
"PCM 44100Hz 16-bit". If you only have an mp3, convert it —
    ffmpeg -i in.mp3 -ar 44100 -ac 1 -sample_fmt s16 out.wav

Usage:
    from audio_responses import VoiceBank

    voice = VoiceBank(ui=ui)              # ui is optional — dims the
                                           # waveform to "speaking" while
                                           # the clip plays, same as TTS
    voice.welcome("Joseph")               # plays "Welcome," + "Joseph"
    voice.play("face_unclear")
    voice.play("unknown_denied")

If a file hasn't been recorded yet, VoiceBank prints what it *would*
have said (or falls back to a Piper speaker, if you pass one in) so
nothing crashes mid-demo while you're still recording lines.
"""

import os
import threading
import wave
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    _SD_AVAILABLE = False


VOICE_DIR  = "assets/voice"
PHRASE_DIR = os.path.join(VOICE_DIR, "phrases")
NAMES_DIR  = os.path.join(VOICE_DIR, "names")

# key -> filename inside assets/voice/phrases/
# Keep this in sync with README_VOICE_AND_CONTACTS.md — that file has
# the exact wording to feed ElevenLabs for each key.
PHRASES = {
    # required — these are the four lines you asked for
    "welcome_prefix":        "welcome_prefix.wav",
    "face_unclear":          "face_unclear.wav",
    "low_confidence_denied": "low_confidence_denied.wav",
    "unknown_denied":        "unknown_denied.wav",
    # optional / recommended extras — record these whenever you like,
    # the system works fine without them (falls back to print/TTS)
    "wake_ack":    "wake_ack.wav",
    "alert_sent":  "alert_sent.wav",
    "goodbye":     "goodbye.wav",
    "emergency_call_message": "emergency_call_message.wav",
}


def _load_wav(path: str):
    with wave.open(path, "rb") as wf:
        n  = wf.getnframes()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
    if sw != 2:
        raise ValueError(
            f"{path}: only 16-bit PCM wav is supported (got {sw*8}-bit). "
            f"Re-export from ElevenLabs as PCM 44100Hz 16-bit."
        )
    data = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        data = data.reshape(-1, ch)
    return data, sr


class VoiceBank:
    """Loads and plays Nova's pre-recorded voice lines."""

    def __init__(self, ui=None, voice_dir: str = VOICE_DIR,
                 fallback_speaker=None):
        """
        ui               : optional NovaWebUI/NovaUI instance — toggles
                            the waveform/orb to 'speaking' while a clip
                            plays, exactly like live TTS does.
        fallback_speaker : optional object with a .say(text) method
                            (e.g. your Piper Speaker) used when a line
                            hasn't been recorded yet.
        """
        self.ui = ui
        self.phrase_dir = os.path.join(voice_dir, "phrases")
        self.names_dir  = os.path.join(voice_dir, "names")
        self._fallback  = fallback_speaker
        self._lock = threading.Lock()

    # ── low level ──────────────────────────────────────────────────────────

    def _play_file(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        if not _SD_AVAILABLE:
            print(f"[VoiceBank] (no audio output available) would play: {path}")
            return True
        data, sr = _load_wav(path)
        if self.ui:
            self.ui.set_speaking(True)
        try:
            sd.play(data, samplerate=sr)
            sd.wait()
        finally:
            if self.ui:
                self.ui.set_speaking(False)
        return True

    # ── public API ─────────────────────────────────────────────────────────

    def play(self, key: str, text_if_missing: Optional[str] = None,
              blocking: bool = True):
        """Play a phrase by key. Set blocking=False to fire-and-forget
        on a background thread (useful from inside a hot loop)."""
        if not blocking:
            threading.Thread(target=self.play, args=(key, text_if_missing, True),
                              daemon=True).start()
            return

        filename = PHRASES.get(key)
        if not filename:
            print(f"[VoiceBank] Unknown phrase key: {key}")
            return

        path = os.path.join(self.phrase_dir, filename)
        with self._lock:
            if self._play_file(path):
                return

        # Not recorded yet — degrade gracefully
        if self._fallback:
            self._fallback.say(text_if_missing or key)
        else:
            print(f"[VoiceBank] Missing {path} — Nova would have said: "
                  f"{text_if_missing or key!r}")

    def welcome(self, first_name: str, blocking: bool = True):
        """Plays 'Welcome,' followed by that person's own recorded name
        clip (assets/voice/names/<first_name>.wav)."""
        if not blocking:
            threading.Thread(target=self.welcome, args=(first_name, True),
                              daemon=True).start()
            return

        with self._lock:
            self._play_file(os.path.join(self.phrase_dir,
                                          PHRASES["welcome_prefix"]))
            name_path = os.path.join(self.names_dir,
                                      f"{first_name.strip().lower()}.wav")
            if not self._play_file(name_path):
                if self._fallback:
                    self._fallback.say(first_name)
                else:
                    print(f"[VoiceBank] Missing name clip: {name_path} "
                          f"— record assets/voice/names/"
                          f"{first_name.strip().lower()}.wav")


if __name__ == "__main__":
    # Quick manifest check: shows which files are recorded / missing
    voice = VoiceBank()
    print("Phrases:")
    for key, fname in PHRASES.items():
        path = os.path.join(PHRASE_DIR, fname)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{status:7s}] {path}")
    print(f"\nNames dir: {NAMES_DIR} "
          f"({'exists' if os.path.isdir(NAMES_DIR) else 'missing'})")

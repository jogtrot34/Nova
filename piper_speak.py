"""
piper_speak.py

Same idea as your voice_trial.py, fixed to use relative paths (like
PIPER_MODEL / PIPER_CONFIG already in main.py) instead of hardcoded
absolute Windows paths, and sounddevice's auto-detected output device
instead of assuming whatever device happened to be default on your
machine.

Standalone — not imported by main.py or anything else yet.

Usage:
    from piper_speak import speak
    speak("Elisa is a stupid boy")

Or from the command line:
    python3 piper_speak.py "Hello, this is a test"
"""

import sys

from audio_devices import configure_audio_device

MODEL_PATH  = "models/en_US-amy-medium.onnx"
CONFIG_PATH = "models/en_US-amy-medium.onnx.json"

_voice = None


def _load_voice():
    global _voice
    if _voice is None:
        from piper import PiperVoice
        _voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)
    return _voice


def speak(text: str):
    """Synthesizes `text` with Piper and plays it through the
    auto-detected output device."""
    import sounddevice as sd
    configure_audio_device()

    voice = _load_voice()
    for chunk in voice.synthesize(text):
        sd.play(chunk.audio_float_array, samplerate=chunk.sample_rate)
        sd.wait()


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "This is a test of Piper text to speech."
    print(f"Speaking: {text!r}")
    speak(text)
    print("Done.")

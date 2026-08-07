import sys

from audio_devices import configure_audio_device

MODEL_PATH = "models/en_US-amy-medium.onnx"
CONFIG_PATH = "models/en_US-amy-medium.onnx.json"

_voice = None

def _load_voice():
    global _voice
    if _voice is None:
        from piper import PiperVoice
        _voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)
    return _voice

def speak(text: str):
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

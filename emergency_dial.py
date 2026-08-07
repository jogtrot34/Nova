import os
import wave
from typing import Optional

import numpy as np

from modem import SimpleModem

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    _SD_AVAILABLE = False


def _modem_audio_device() -> Optional[int]:
    env = os.environ.get("NOVA_MODEM_AUDIO_DEVICE")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            print(f"[EmergencyDial] NOVA_MODEM_AUDIO_DEVICE={env!r} isn't "
                  f"a valid device index — using the system default output.")
    return None   # None = sounddevice's current default output device


def _load_wav(path: str):
    with wave.open(path, "rb") as wf:
        n  = wf.getnframes()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: only 16-bit PCM wav is supported")
    data = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        data = data.reshape(-1, ch)
    return data, sr


class EmergencyDialer:
    """Calls a number and, if answered, plays a pre-recorded WAV into
    the call before hanging up. Degrades gracefully — if the modem
    audio device isn't configured or found, the call still completes
    and the outcome is still reported, it just won't play anything."""

    def __init__(self, port: Optional[str] = None):
        self.modem = SimpleModem()
        self._connected = False
        self._port = port

    def connect(self) -> bool:
        self._connected = self.modem.connect(self._port)
        return self._connected

    def call(self, number: str, timeout: int = 30) -> str:
        """Returns 'answered' | 'denied' | 'busy' | 'no_response' | 'error'."""
        if not self._connected and not self.connect():
            return "error"
        return self.modem.make_call(number, timeout=timeout)

    # Alias so EmergencyDialer is a drop-in for anything (like
    # notifier.Notifier) that expects the raw SimpleModem/ModemController
    # interface, which calls .make_call() rather than .call().
    def make_call(self, number: str, timeout: int = 30, **_ignored) -> str:
        return self.call(number, timeout=timeout)

    def call_and_play(self, number: str, wav_path: str,
                       timeout: int = 30) -> dict:
        """
        Calls `number`; if answered, plays `wav_path` into the call
        (once) before hanging up.

        Returns:
            {"status": "answered"|"denied"|"busy"|"no_response"|"error",
             "played": bool, "error": str or None}
        """
        result = {"status": None, "played": False, "error": None}

        if not self._connected and not self.connect():
            result["status"] = "error"
            result["error"]  = "Could not connect to modem"
            return result

        status = self.modem.make_call(number, timeout=timeout,
                                       hangup_on_answer=False)
        result["status"] = status

        if status != "answered":
            return result

        try:
            if not _SD_AVAILABLE:
                result["error"] = ("call connected, but sounddevice isn't "
                                    "available — nothing was played")
                return result
            if not os.path.exists(wav_path):
                result["error"] = f"Message file not found: {wav_path}"
                return result

            data, sr = _load_wav(wav_path)
            device = _modem_audio_device()
            sd.play(data, samplerate=sr, device=device)
            sd.wait()
            result["played"] = True
        except Exception as e:
            result["error"] = f"Could not play message into the call: {e}"
        finally:
            try:
                self.modem.hangup()
            except Exception:
                pass

        return result

    def send_sms(self, number: str, message: str) -> bool:
        if not self._connected and not self.connect():
            return False
        return self.modem.send_sms(number, message)

    def disconnect(self):
        self.modem.disconnect()
        self._connected = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the modem directly from the terminal")
    parser.add_argument("--port", default=None, help="e.g. /dev/ttyACM0 — auto-detects if omitted")
    parser.add_argument("--call", metavar="NUMBER", help="Just call and report the result")
    parser.add_argument("--sms", nargs=2, metavar=("NUMBER", "MESSAGE"))
    parser.add_argument("--call-and-play", nargs=2, metavar=("NUMBER", "WAV_PATH"),
                        help="Call, and if answered, play that wav into the call")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    dialer = EmergencyDialer(port=args.port)

    if args.call:
        print("Connecting...")
        status = dialer.call(args.call, timeout=args.timeout)
        print(f"Result: {status}")
    elif args.sms:
        number, message = args.sms
        ok = dialer.send_sms(number, message)
        print(f"SMS {'sent' if ok else 'failed'}")
    elif args.call_and_play:
        number, wav_path = args.call_and_play
        result = dialer.call_and_play(number, wav_path, timeout=args.timeout)
        print(f"Result: {result}")
    else:
        parser.print_help()

    dialer.disconnect()

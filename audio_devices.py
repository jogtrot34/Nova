"""
audio_devices.py

Nova used to hardcode `sd.default.device = (1, 1)` with a comment
saying "ALC236 Analog on Zorin" — that was the input/output device
index on the machine this was originally built on. On any other
machine, index 1 might be a completely different piece of hardware
(an HDMI output with zero input channels, a monitor device, whatever
your system happens to enumerate there), which is exactly what
produces:

    PortAudioError: Error opening InputStream: Invalid number of
    channels [PaErrorCode -9998]

This module picks a device that actually has the channels it needs,
instead of assuming index 1 is always right. Call
configure_audio_device() once, early, before anything else touches
sounddevice — device selection is global process state in
PortAudio/sounddevice, so this one call is enough for every module
(voice_layer.py, web/server.py's MicMonitor, audio_responses.py) to
inherit a working default.

Override without touching any code, if auto-detection ever picks the
wrong device on your machine:

    NOVA_INPUT_DEVICE=3 NOVA_OUTPUT_DEVICE=3 python3 main.py

See what's actually available on this machine:

    python3 audio_devices.py
"""

import os


def _find_device(kind: str):
    """kind is 'input' or 'output'. Returns a device index or None."""
    import sounddevice as sd

    channels_attr = "max_input_channels" if kind == "input" else "max_output_channels"
    default_slot  = 0 if kind == "input" else 1

    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[AudioDevices] Could not query audio devices: {e}")
        return None

    # 1. Explicit override via environment variable always wins.
    env_var  = f"NOVA_{kind.upper()}_DEVICE"
    override = os.environ.get(env_var)
    if override is not None:
        try:
            idx = int(override)
            if devices[idx][channels_attr] >= 1:
                return idx
            print(f"[AudioDevices] {env_var}={override} has no {kind} "
                  f"channels ({devices[idx]['name']}) — ignoring override.")
        except (ValueError, IndexError):
            print(f"[AudioDevices] {env_var}={override} is not a valid "
                  f"device index (0-{len(devices)-1}) — ignoring.")

    # 2. Try whatever the OS/PortAudio already considers the default.
    try:
        default_idx = sd.default.device[default_slot]
        if default_idx is not None and default_idx >= 0:
            if devices[default_idx][channels_attr] >= 1:
                return default_idx
    except Exception:
        pass

    # 3. Otherwise, scan for the first device that actually has the
    #    channels this direction needs.
    for idx, d in enumerate(devices):
        if d[channels_attr] >= 1:
            return idx

    return None


def configure_audio_device():
    """
    Picks a working input device and a working output device and sets
    them as the sounddevice defaults for the whole process.

    Safe to call more than once. Safe to call on a machine with no
    audio hardware at all — sd.default.device just ends up (None, None)
    and callers that need real audio (recording, TTS playback) will
    get a clear error at that point instead of this function crashing.
    """
    import sounddevice as sd

    in_idx  = _find_device("input")
    out_idx = _find_device("output")
    sd.default.device = (in_idx, out_idx)

    def _name(idx):
        if idx is None:
            return "(none found)"
        try:
            return sd.query_devices()[idx]["name"]
        except Exception:
            return f"#{idx}"

    print(f"[AudioDevices] input={_name(in_idx)!r}  output={_name(out_idx)!r}")
    return in_idx, out_idx


def test_mic(device=None, seconds: float = 3.0):
    """
    Streams from `device` (or the current default input if None) for a
    few seconds and prints a live level meter, so you can see whether
    it's actually picking up sound before trusting it for enrollment.
    """
    import sounddevice as sd
    import numpy as np
    import time as _time

    devices = sd.query_devices()
    if device is None:
        device = sd.default.device[0]
    if device is None:
        print("[AudioDevices] No input device selected — nothing to test.")
        return False

    name = devices[device]["name"] if 0 <= device < len(devices) else "?"
    print(f"[AudioDevices] Testing device {device} ({name}) "
          f"for {seconds:.0f}s — make some noise (talk, tap the mic)...")

    levels = []

    def cb(indata, frames, time_info, status):
        rms = float(np.sqrt(np.mean(indata ** 2)))
        levels.append(rms)
        bar = "#" * int(min(rms * 250, 40))
        print(f"\r  level: {bar:<40} {rms:.4f}", end="", flush=True)

    try:
        with sd.InputStream(device=device, channels=1, samplerate=44100,
                             dtype="float32", callback=cb):
            _time.sleep(seconds)
    except Exception as e:
        print(f"\n[AudioDevices] Failed to open device {device}: {e}")
        return False

    print()
    peak = max(levels) if levels else 0.0
    if peak > 0.01:
        print(f"[AudioDevices] Alive — peak level {peak:.4f}. "
              f"This device works: NOVA_INPUT_DEVICE={device}")
        return True
    else:
        print(f"[AudioDevices] No meaningful signal (peak {peak:.4f}) — "
              f"opened fine but heard nothing. Try another device, or "
              f"check it isn't muted.")
        return False


def test_all_input_devices(seconds: float = 2.5):
    """Runs test_mic() against every device that reports input channels,
    so you don't have to guess which index is the real microphone."""
    import sounddevice as sd
    devices = sd.query_devices()
    candidates = [i for i, d in enumerate(devices)
                  if d["max_input_channels"] >= 1]
    if not candidates:
        print("[AudioDevices] No input-capable devices found at all.")
        return
    print(f"[AudioDevices] {len(candidates)} input-capable device(s) found.\n")
    working = []
    for idx in candidates:
        if test_mic(device=idx, seconds=seconds):
            working.append(idx)
        print()
    if working:
        print(f"[AudioDevices] Working device(s): {working}")
        print(f"[AudioDevices] Recommended: "
              f"NOVA_INPUT_DEVICE={working[0]}")
    else:
        print("[AudioDevices] None of them picked up sound. Check your "
              "system's sound settings (input muted? wrong device "
              "selected at the OS level?) and try again.")


if __name__ == "__main__":
    import sys
    import sounddevice as sd

    if "--test-mic" in sys.argv:
        idx = None
        for arg in sys.argv[sys.argv.index("--test-mic") + 1:]:
            if arg.isdigit():
                idx = int(arg)
                break
        configure_audio_device()
        test_mic(device=idx)
    elif "--test-all" in sys.argv:
        configure_audio_device()
        test_all_input_devices()
    else:
        print(sd.query_devices())
        print()
        configure_audio_device()
        print()
        print("Tip: 'python3 audio_devices.py --test-mic [index]' or "
              "'--test-all' to check which device actually picks up sound.")

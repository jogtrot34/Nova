"""
notification_settings.py

Small persisted store for the three notification "modes" the web UI
exposes as toggle switches:

    log_digest_enabled            — text emergency contact #1 a summary
                                     of identification activity every
                                     `log_digest_interval_minutes`
                                     minutes, but ONLY if anything
                                     actually happened in that window.
    unknown_text_enabled          — text emergency contact #1 the
                                     moment an unrecognised person is
                                     detected.
    unknown_call_enabled          — place a call to emergency contact
                                     #1 if that unrecognised person is
                                     still there a little while later
                                     (not on the very first frame —
                                     that would dial out on every
                                     passing stranger).

Stored at data/notification_settings.json — same "data/" convention as
data/nova.db and data/voiceprints. Safe to import from anywhere; reads
are cheap (small dict), writes go through a lock and are re-read from
disk each time set() is called so multiple processes/threads never
silently disagree with each other for long.

Usage:
    from notification_settings import NotificationSettings
    settings = NotificationSettings()
    settings.get_all()                       # -> dict
    settings.set("unknown_text_enabled", True)
"""

import json
import os
import threading
from pathlib import Path

SETTINGS_PATH = "data/notification_settings.json"

DEFAULTS = {
    "log_digest_enabled":          False,
    "log_digest_interval_minutes": 10,
    "unknown_text_enabled":        False,
    "unknown_call_enabled":        False,
    # How long (seconds) an unrecognised person must stay in frame
    # before unknown_call_enabled escalates from "texted" to "called".
    "unknown_call_after_seconds":  15,
}


class NotificationSettings:
    def __init__(self, path: str = SETTINGS_PATH):
        self.path = path
        self._lock = threading.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path):
            self._write(dict(DEFAULTS))

    def _read(self) -> dict:
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
        # Backfill any keys added in later versions without clobbering
        # what's already been saved.
        merged = dict(DEFAULTS)
        merged.update(data)
        return merged

    def _write(self, data: dict):
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def get_all(self) -> dict:
        with self._lock:
            return self._read()

    def get(self, key: str):
        return self.get_all().get(key, DEFAULTS.get(key))

    def set(self, key: str, value) -> dict:
        if key not in DEFAULTS:
            raise KeyError(f"Unknown setting: {key}")
        with self._lock:
            data = self._read()
            # Keep types sane regardless of what the frontend sends
            if isinstance(DEFAULTS[key], bool):
                value = bool(value)
            elif isinstance(DEFAULTS[key], (int, float)):
                try:
                    value = type(DEFAULTS[key])(value)
                except (TypeError, ValueError):
                    value = DEFAULTS[key]
            data[key] = value
            self._write(data)
            return data

    def update(self, patch: dict) -> dict:
        """Apply several settings at once — used by the one PATCH-style
        endpoint the web UI calls when you flip a switch."""
        with self._lock:
            data = self._read()
            for key, value in patch.items():
                if key not in DEFAULTS:
                    continue
                if isinstance(DEFAULTS[key], bool):
                    value = bool(value)
                elif isinstance(DEFAULTS[key], (int, float)):
                    try:
                        value = type(DEFAULTS[key])(value)
                    except (TypeError, ValueError):
                        continue
                data[key] = value
            self._write(data)
            return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"))
    args = parser.parse_args()

    s = NotificationSettings()
    if args.set:
        key, value = args.set
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        print(s.set(key, value))
    else:
        print(json.dumps(s.get_all(), indent=2))

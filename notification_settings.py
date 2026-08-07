import json
import os
import threading
from pathlib import Path

SETTINGS_PATH = "data/notification_settings.json"

DEFAULTS = {
    "log_digest_enabled": False,
    "log_digest_interval_minutes": 10,
    "unknown_text_enabled": False,
    "unknown_call_enabled": False,
    "unknown_call_after_seconds": 15,
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

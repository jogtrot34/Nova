"""
camera_devices.py

Same problem class as the mic issue, same fix: main.py assumed camera
index 0 was always right and crashed the whole app (sys.exit(1)) if it
wasn't. Some camera indices also "open" successfully per the driver
and then never actually deliver a frame — this checks for a real
frame, not just cv2.VideoCapture().isOpened().

    python3 camera_devices.py            # scan indices 0-5, report which work
    NOVA_CAMERA_INDEX=2 python3 main.py  # pin it explicitly, no code changes
"""

import os
import time


def _try_index(idx, warmup_frames=5, timeout=3.0):
    import cv2
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return False
    deadline = time.time() + timeout
    got_frame = False
    for _ in range(warmup_frames):
        if time.time() > deadline:
            break
        ret, frame = cap.read()
        if ret and frame is not None:
            got_frame = True
            break
    cap.release()
    return got_frame


def find_working_camera(max_index: int = 6):
    """Returns a working camera index, or None if nothing on this
    machine actually delivers a frame."""
    override = os.environ.get("NOVA_CAMERA_INDEX")
    if override is not None:
        try:
            idx = int(override)
            if _try_index(idx):
                return idx
            print(f"[CameraDevices] NOVA_CAMERA_INDEX={override} did not "
                  f"deliver a frame — falling back to auto-detect.")
        except ValueError:
            print(f"[CameraDevices] NOVA_CAMERA_INDEX={override} isn't a "
                  f"valid index — falling back to auto-detect.")

    for idx in range(max_index):
        if _try_index(idx):
            return idx
    return None


if __name__ == "__main__":
    print("Scanning camera indices 0-5 (this can take a few seconds "
          "per index)...")
    for idx in range(6):
        ok = _try_index(idx)
        print(f"  index {idx}: {'OK — delivers frames' if ok else 'no signal'}")

    working = find_working_camera()
    print()
    if working is not None:
        print(f"Recommended: NOVA_CAMERA_INDEX={working}")
    else:
        print("No working camera found. Check it's plugged in and not "
              "held open by another app (close any other camera app, "
              "browser tab using the camera, etc).")

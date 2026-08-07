"""
reset_nova.py

Wipes Nova completely: every person, clothing entry, and access log
row in the database, every saved voiceprint (.npy) file, and —
optionally — the known_faces/ photo and video folders too.

    python3 reset_nova.py               # DB + voiceprints only
    python3 reset_nova.py --with-photos # also deletes known_faces/*

This is destructive and asks for confirmation before doing anything.
For just clearing the database rows (leaving all your enrollment
photos/videos/voice files untouched so you can re-run enrollment
against them), use:

    python3 db.py --reset
"""

import argparse
import glob
import os
import shutil

from db import NovaDB, DB_PATH
from voice_layer import VOICE_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-photos", action="store_true",
                        help="Also delete known_faces/ (every enrollment "
                             "photo/video). Off by default so you can "
                             "re-enroll from the same source material.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation prompt (for scripts)")
    args = parser.parse_args()

    print("This will permanently delete:")
    print(f"  - every row in {DB_PATH} (people, clothing, access log)")
    print(f"  - every voiceprint file in {VOICE_DIR}/*.npy")
    if args.with_photos:
        print("  - every photo/video in known_faces/*")
    print()

    if not args.yes:
        confirm = input("Type YES to confirm: ").strip()
        if confirm != "YES":
            print("Cancelled — nothing was deleted.")
            return

    db = NovaDB(DB_PATH)
    db.reset_all()

    npy_files = glob.glob(os.path.join(VOICE_DIR, "*.npy"))
    for f in npy_files:
        os.remove(f)
    print(f"[Reset] Removed {len(npy_files)} voiceprint file(s) "
          f"from {VOICE_DIR}/")

    if args.with_photos and os.path.isdir("known_faces"):
        for entry in os.listdir("known_faces"):
            path = os.path.join("known_faces", entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        print("[Reset] Cleared known_faces/")

    print("\n[Reset] Done. Nova is back to a clean slate.")


if __name__ == "__main__":
    main()

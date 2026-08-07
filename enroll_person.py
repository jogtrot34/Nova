"""
enroll_person.py

One script to fully register a new person into Nova.

Steps:
  1. Enter their name, role, and folder path
  2. Face encodings extracted from photos/videos in that folder
  3. Voice embeddings extracted from every audio file in that same
     folder (or a separate --voice-folder) — no live mic recording
     needed if you already have samples on disk. Falls back to live
     recording only if you ask for it or no audio files are found.
  4. Done -- they are in the DB and ready for identification

A person's folder can mix everything together:
    known_faces/joseph/
        photo1.jpg
        photo2.jpg
        clip.mp4
        sample1.wav
        sample2.wav
        ...as many voice samples as you've got — more variety
        (different sentences, sessions, moods) makes a stronger
        voiceprint.

Usage:
    python3 enroll_person.py

Or non-interactively:
    python3 enroll_person.py --first Joseph --last Wella \
        --role owner --access full --folder known_faces/joseph

    # voice samples in a different folder than the photos:
    python3 enroll_person.py --first Joseph --last Wella \
        --folder known_faces/joseph --voice-folder voice_samples/joseph

    # top up an existing person's voiceprint with a fresh batch:
    python3 enroll_person.py --append-voice --first Joseph --last Wella \
        --folder known_faces/joseph --skip-face
"""

import argparse
import os
import sys
from db import NovaDB
from face_layer import enroll_person_faces
from voice_layer import enroll_person_voice, enroll_person_voice_from_folder


def interactive_enroll(db: NovaDB):
    print("=" * 55)
    print("  Nova -- Person Enrollment")
    print("=" * 55)
    print()

    first = input("First name: ").strip()
    last  = input("Last name : ").strip()

    if not first or not last:
        print("[ERROR] Name is required.")
        sys.exit(1)

    print()
    print("Roles: owner / staff / visitor / guest")
    role  = input("Role      : ").strip() or "visitor"

    print("Access levels: full / limited / none")
    access = input("Access    : ").strip() or "none"

    notes  = input("Notes (beard, height, anything useful): ").strip()

    print()
    print("Photo/video folder (e.g. known_faces/joseph)")
    print("Leave blank to skip face enrollment and do it later.")
    folder = input("Folder    : ").strip()

    print()
    pid = db.add_person(first, last, role=role,
                        access_level=access, notes=notes)
    print(f"\n[OK] Registered {first} {last} (id={pid})")

    # Face enrollment
    if folder:
        if not os.path.exists(folder):
            print(f"[WARN] Folder not found: {folder}")
            print("       Create it, add photos/videos, then run:")
            print(f"       python3 face_layer.py enroll {pid}")
        else:
            # Copy/link folder to known_faces/<firstname>
            dest_dir = os.path.join("known_faces", first.lower())
            if os.path.abspath(folder) != os.path.abspath(dest_dir):
                print(f"[INFO] Using folder: {folder}")

            # Temporarily patch DATA_DIR for enroll function
            import face_layer as fl
            orig = fl.DATA_DIR if hasattr(fl, "DATA_DIR") else "known_faces"

            n = enroll_person_faces(pid, db, data_dir=os.path.dirname(
                os.path.abspath(folder)))
            if n == 0:
                print("[WARN] No face encodings saved. Check that photos")
                print("       contain clear, well-lit faces.")
            else:
                print(f"[OK] {n} face encoding(s) saved.")
    else:
        print("[SKIP] Face enrollment skipped.")
        print(f"       Run later: python3 face_layer.py enroll {pid}")

    # Voice enrollment
    print()
    voice_folder = folder if folder and os.path.isdir(folder) else None
    if voice_folder:
        n = enroll_person_voice_from_folder(pid, db, voice_folder)
        if n > 0:
            print(f"[OK] Voice enrolled from {n} sample(s) found in the folder.")
        else:
            print(f"[INFO] No audio files found in {voice_folder}.")
            do_voice = input("Record voice live instead? (y/n) [y]: ").strip().lower()
            if do_voice in ("", "y", "yes"):
                enroll_person_voice(pid, db)
            else:
                print("[SKIP] Voice enrollment skipped.")
                print(f"       Run later: python3 voice_layer.py "
                      f"enroll-folder {pid} <folder-of-audio-files>")
    else:
        do_voice = input("Enroll voice now (live recording)? (y/n) [y]: ").strip().lower()
        if do_voice in ("", "y", "yes"):
            enroll_person_voice(pid, db)
        else:
            print("[SKIP] Voice enrollment skipped.")
            print(f"       Run later: python3 voice_layer.py "
                  f"enroll-folder {pid} <folder-of-audio-files>")

    print()
    print("=" * 55)
    print(f"  Enrollment complete for {first} {last}")
    db.summary()


def cli_enroll(db: NovaDB, args):
    pid = db.add_person(
        args.first, args.last,
        role=args.role, access_level=args.access,
        notes=args.notes or "",
    )
    print(f"[OK] Registered {args.first} {args.last} (id={pid})")

    if args.folder and not args.skip_face:
        n = enroll_person_faces(pid, db,
                                 data_dir=os.path.dirname(
                                     os.path.abspath(args.folder)))
        print(f"[OK] {n} face encoding(s) saved.")

    if not args.skip_voice:
        voice_folder = args.voice_folder or args.folder
        n = 0
        if voice_folder:
            n = enroll_person_voice_from_folder(
                pid, db, voice_folder, append=args.append_voice)
        if n == 0:
            if args.live_voice:
                enroll_person_voice(pid, db)
            else:
                where = voice_folder or "(no folder given)"
                print(f"[WARN] No voice samples found in {where}.")
                print(f"       Add audio files there, or run: "
                      f"python3 voice_layer.py enroll-folder {pid} <folder>")
                print(f"       Or pass --live-voice to record live instead.")

    db.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enroll a new person into Nova")
    parser.add_argument("--first",      help="First name")
    parser.add_argument("--last",       help="Last name")
    parser.add_argument("--role",       default="visitor")
    parser.add_argument("--access",     default="none")
    parser.add_argument("--notes",      default="")
    parser.add_argument("--folder",     help="Path to photo/video/audio folder")
    parser.add_argument("--voice-folder",
                        help="Folder of voice samples, if different from "
                             "--folder")
    parser.add_argument("--skip-face",  action="store_true",
                        help="Don't touch face enrollment")
    parser.add_argument("--skip-voice", action="store_true",
                        help="Don't touch voice enrollment")
    parser.add_argument("--live-voice", action="store_true",
                        help="Fall back to live mic recording if no audio "
                             "files are found in the voice folder")
    parser.add_argument("--append-voice", action="store_true",
                        help="Add to this person's existing voiceprint "
                             "instead of replacing it")

    args = parser.parse_args()
    db   = NovaDB()

    if args.first and args.last:
        cli_enroll(db, args)
    else:
        interactive_enroll(db)

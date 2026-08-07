"""
db.py

Nova's single source of truth.

All person data lives here — identity, face encodings, voiceprints,
clothing memory, and access logs. Nothing is scattered across loose
files except the raw training images (which stay in known_faces/).

Tables:
    people          — one row per registered person
    clothing_memory — clothing observations linked to a person
    access_log      — every identification event logged here

Usage:
    from db import NovaDB
    db = NovaDB()
    db.add_person("Joseph", "Wella", "owner", "full")
"""

import sqlite3
import os
import json
import numpy as np
from datetime import datetime

DB_PATH = "data/nova.db"


class NovaDB:

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS people (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_name          TEXT NOT NULL,
                    last_name           TEXT NOT NULL,
                    role                TEXT DEFAULT 'visitor',
                    access_level        TEXT DEFAULT 'none',
                    face_encodings      TEXT,
                    voiceprint_path     TEXT,
                    notes               TEXT,
                    registered_at       TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS clothing_memory (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       INTEGER NOT NULL REFERENCES people(id),
                    torso_hist      TEXT NOT NULL,
                    legs_hist       TEXT NOT NULL,
                    shoes_hist      TEXT,
                    label           TEXT,
                    recorded_at     TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS access_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       INTEGER REFERENCES people(id),
                    timestamp       TEXT DEFAULT (datetime('now')),
                    method          TEXT,
                    confidence      REAL,
                    decision        TEXT,
                    notes           TEXT
                );
            """)
            # Migration: pin_hash didn't exist in earlier versions of
            # this schema. ALTER TABLE ADD COLUMN is safe to retry —
            # sqlite just errors "duplicate column" if it's already
            # there, which we ignore.
            try:
                conn.execute("ALTER TABLE people ADD COLUMN pin_hash TEXT")
            except sqlite3.OperationalError:
                pass
        print(f"[DB] Ready at {self.db_path}")

    # ── People ────────────────────────────────────────────────────────────────

    def add_person(self, first_name: str, last_name: str,
                   role: str = "visitor", access_level: str = "none",
                   notes: str = "") -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO people
                   (first_name, last_name, role, access_level, notes)
                   VALUES (?, ?, ?, ?, ?)""",
                (first_name.strip(), last_name.strip(),
                 role.strip(), access_level.strip(), notes.strip())
            )
            person_id = cur.lastrowid
        print(f"[DB] Registered: {first_name} {last_name} "
              f"(id={person_id}, role={role}, access={access_level})")
        return person_id

    def get_person(self, person_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM people WHERE id = ?", (person_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_person_by_name(self, first_name: str,
                            last_name: str = "") -> dict | None:
        with self._connect() as conn:
            if last_name:
                row = conn.execute(
                    "SELECT * FROM people "
                    "WHERE LOWER(first_name)=? AND LOWER(last_name)=?",
                    (first_name.lower(), last_name.lower())
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM people WHERE LOWER(first_name)=?",
                    (first_name.lower(),)
                ).fetchone()
        return dict(row) if row else None

    def list_people(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM people ORDER BY first_name"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_person(self, person_id: int, **kwargs):
        allowed = {"first_name", "last_name", "role",
                   "access_level", "notes"}
        fields  = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values     = list(fields.values()) + [person_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE people SET {set_clause} WHERE id = ?", values
            )

    def delete_person(self, person_id: int):
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM clothing_memory WHERE person_id = ?",
                (person_id,))
            conn.execute(
                "DELETE FROM people WHERE id = ?", (person_id,))
        print(f"[DB] Deleted person id={person_id} and their clothing memory.")

    # ── PIN (fallback confirmation for borderline face/voice matches) ─────────

    def set_person_pin(self, person_id: int, pin: str):
        """Stores a salted hash of the PIN — never the PIN itself."""
        import hashlib
        salt   = os.urandom(16).hex()
        digest = hashlib.pbkdf2_hmac(
            "sha256", pin.encode(), bytes.fromhex(salt), 100_000).hex()
        with self._connect() as conn:
            conn.execute(
                "UPDATE people SET pin_hash = ? WHERE id = ?",
                (f"{salt}:{digest}", person_id))
        print(f"[DB] PIN set for person id={person_id}")

    def verify_person_pin(self, person_id: int, pin: str) -> bool:
        import hashlib
        person = self.get_person(person_id)
        if not person or not person.get("pin_hash"):
            return False
        salt, digest = person["pin_hash"].split(":", 1)
        check = hashlib.pbkdf2_hmac(
            "sha256", pin.encode(), bytes.fromhex(salt), 100_000).hex()
        return check == digest

    def person_has_pin(self, person_id: int) -> bool:
        person = self.get_person(person_id)
        return bool(person and person.get("pin_hash"))

    def remove_person_pin(self, person_id: int):
        with self._connect() as conn:
            conn.execute(
                "UPDATE people SET pin_hash = NULL WHERE id = ?",
                (person_id,))

    # ── Face encodings ────────────────────────────────────────────────────────

    def save_face_encodings(self, person_id: int,
                             encodings: list[np.ndarray]):
        serialised = json.dumps([e.tolist() for e in encodings])
        with self._connect() as conn:
            conn.execute(
                "UPDATE people SET face_encodings = ? WHERE id = ?",
                (serialised, person_id)
            )
        print(f"[DB] Saved {len(encodings)} face encoding(s) for id={person_id}")

    def add_face_encodings(self, person_id: int,
                            new_encodings: list[np.ndarray]) -> int:
        """Append to whatever face encodings this person already has,
        instead of overwriting them. Used by the web UI's 'capture
        another sample' flow. Returns the new total count."""
        existing = self.load_face_encodings(person_id)
        combined = existing + list(new_encodings)
        self.save_face_encodings(person_id, combined)
        return len(combined)

    def load_face_encodings(self, person_id: int) -> list[np.ndarray]:
        row = self.get_person(person_id)
        if not row or not row["face_encodings"]:
            return []
        return [np.array(e) for e in json.loads(row["face_encodings"])]

    def load_all_face_encodings(self) -> list[tuple[int, str, np.ndarray]]:
        people = self.list_people()
        result = []
        for p in people:
            if not p["face_encodings"]:
                continue
            name      = f"{p['first_name']} {p['last_name']}".strip()
            encodings = [np.array(e)
                         for e in json.loads(p["face_encodings"])]
            for enc in encodings:
                result.append((p["id"], name, enc))
        return result

    # ── Voiceprint ────────────────────────────────────────────────────────────

    def save_voiceprint_path(self, person_id: int, path: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE people SET voiceprint_path = ? WHERE id = ?",
                (path, person_id)
            )
        print(f"[DB] Voiceprint path saved for id={person_id} -> {path}")

    def load_voiceprint(self, person_id: int) -> np.ndarray | None:
        row = self.get_person(person_id)
        if not row or not row["voiceprint_path"]:
            return None
        path = row["voiceprint_path"]
        if not os.path.exists(path):
            print(f"[DB][WARN] Voiceprint file missing: {path}")
            return None
        return np.load(path)

    def load_all_voiceprints(self) -> list[tuple[int, str, np.ndarray]]:
        people = self.list_people()
        result = []
        for p in people:
            if not p["voiceprint_path"]:
                continue
            vp = self.load_voiceprint(p["id"])
            if vp is not None:
                name = f"{p['first_name']} {p['last_name']}".strip()
                result.append((p["id"], name, vp))
        return result

    # ── Clothing memory ───────────────────────────────────────────────────────

    def add_clothing_entry(self, person_id: int,
                            torso_hist: np.ndarray,
                            legs_hist:  np.ndarray,
                            shoes_hist: np.ndarray | None = None,
                            label: str = ""):
        t = json.dumps(torso_hist.tolist())
        l = json.dumps(legs_hist.tolist())
        s = json.dumps(shoes_hist.tolist()) if shoes_hist is not None else None

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO clothing_memory
                   (person_id, torso_hist, legs_hist, shoes_hist, label)
                   VALUES (?, ?, ?, ?, ?)""",
                (person_id, t, l, s, label)
            )

        # Keep at most 40 entries per person
        with self._connect() as conn:
            conn.execute("""
                DELETE FROM clothing_memory
                WHERE person_id = ? AND id NOT IN (
                    SELECT id FROM clothing_memory
                    WHERE person_id = ?
                    ORDER BY recorded_at DESC
                    LIMIT 40
                )
            """, (person_id, person_id))

    def load_clothing_entries(self, person_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM clothing_memory
                   WHERE person_id = ?
                   ORDER BY recorded_at DESC""",
                (person_id,)
            ).fetchall()

        entries = []
        for row in rows:
            entries.append({
                "torso_hist": np.array(json.loads(row["torso_hist"]),
                                        dtype=np.float32),
                "legs_hist":  np.array(json.loads(row["legs_hist"]),
                                        dtype=np.float32),
                "shoes_hist": np.array(json.loads(row["shoes_hist"]),
                                        dtype=np.float32)
                               if row["shoes_hist"] else None,
                "label":      row["label"],
                "recorded_at":row["recorded_at"],
            })
        return entries

    def load_all_clothing_entries(self) -> list[tuple[int, str, dict]]:
        people = self.list_people()
        result = []
        for p in people:
            name    = f"{p['first_name']} {p['last_name']}".strip()
            entries = self.load_clothing_entries(p["id"])
            for entry in entries:
                result.append((p["id"], name, entry))
        return result

    # ── Access log ────────────────────────────────────────────────────────────

    def log_access(self, person_id: int | None,
                   method: str, confidence: float,
                   decision: str, notes: str = ""):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO access_log
                   (person_id, method, confidence, decision, notes)
                   VALUES (?, ?, ?, ?, ?)""",
                (person_id, method, confidence, decision, notes)
            )

    def get_access_log(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT l.*, p.first_name, p.last_name
                   FROM access_log l
                   LEFT JOIN people p ON l.person_id = p.id
                   ORDER BY l.timestamp DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Utility ───────────────────────────────────────────────────────────────

    def reset_all(self):
        """Wipes every person, clothing entry, and access log row.
        Does NOT touch data/voiceprints/*.npy or known_faces/ — those
        are files on disk, not DB rows. Use reset_nova.py to clear
        everything (DB + those files) in one go."""
        with self._connect() as conn:
            conn.execute("DELETE FROM people")
            conn.execute("DELETE FROM clothing_memory")
            conn.execute("DELETE FROM access_log")
            # also clear the optional contacts tables if contacts_db.py
            # has been used against this same database file
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "emergency_contacts" in tables:
                conn.execute("DELETE FROM emergency_contacts")
            if "person_contacts" in tables:
                conn.execute("DELETE FROM person_contacts")
            # restart auto-increment ids from 1
            conn.execute("DELETE FROM sqlite_sequence")
        print(f"[DB] Reset — {self.db_path} is now empty.")

    def summary(self):
        with self._connect() as conn:
            n_people   = conn.execute(
                "SELECT COUNT(*) FROM people").fetchone()[0]
            n_clothing = conn.execute(
                "SELECT COUNT(*) FROM clothing_memory").fetchone()[0]
            n_logs     = conn.execute(
                "SELECT COUNT(*) FROM access_log").fetchone()[0]
            n_face     = conn.execute(
                "SELECT COUNT(*) FROM people "
                "WHERE face_encodings IS NOT NULL").fetchone()[0]
            n_voice    = conn.execute(
                "SELECT COUNT(*) FROM people "
                "WHERE voiceprint_path IS NOT NULL").fetchone()[0]

        print("=" * 45)
        print("  Nova Database Summary")
        print("=" * 45)
        print(f"  People registered : {n_people}")
        print(f"  With face data    : {n_face}")
        print(f"  With voice data   : {n_voice}")
        print(f"  Clothing entries  : {n_clothing}")
        print(f"  Access log entries: {n_logs}")
        print("=" * 45)
        people = self.list_people()
        for p in people:
            print(f"  [{p['id']}] {p['first_name']} {p['last_name']}"
                  f" — {p['role']} ({p['access_level']} access)")
        print("=" * 45)


# ── Standalone test / CLI ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nova database — inspect and manage from the terminal")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe every person, clothing entry, and access log row")
    parser.add_argument("--add-person", nargs=2, metavar=("FIRST", "LAST"),
                        help="Quickly register someone, e.g. --add-person Joseph Wella")
    parser.add_argument("--role", default="visitor")
    parser.add_argument("--access", default="none")
    parser.add_argument("--list-people", action="store_true",
                        help="Print every registered person with their id")
    parser.add_argument("--set-pin", nargs=2, metavar=("PERSON_ID", "PIN"),
                        help="--set-pin 3 4821")
    parser.add_argument("--verify-pin", nargs=2, metavar=("PERSON_ID", "PIN"),
                        help="--verify-pin 3 4821 — prints whether it matches")
    parser.add_argument("--remove-pin", type=int, metavar="PERSON_ID")
    args = parser.parse_args()

    db = NovaDB()
    did_something = False

    if args.reset:
        did_something = True
        print("This deletes every person, clothing entry, and access "
              "log row. Face/voice files on disk are left alone — use "
              "reset_nova.py to wipe those too.")
        confirm = input("Type YES to confirm: ").strip()
        if confirm == "YES":
            db.reset_all()
        else:
            print("Cancelled — nothing was deleted.")

    if args.add_person:
        did_something = True
        first, last = args.add_person
        pid = db.add_person(first, last, role=args.role, access_level=args.access)
        print(f"[DB] id={pid}")

    if args.list_people:
        did_something = True
        for p in db.list_people():
            pin_tag = " [pin set]" if p.get("pin_hash") else ""
            print(f"  [{p['id']}] {p['first_name']} {p['last_name']} "
                  f"— {p['role']} ({p['access_level']}){pin_tag}")

    if args.set_pin:
        did_something = True
        pid, pin = args.set_pin
        db.set_person_pin(int(pid), pin)

    if args.verify_pin:
        did_something = True
        pid, pin = args.verify_pin
        ok = db.verify_person_pin(int(pid), pin)
        print(f"[DB] PIN {'MATCHES' if ok else 'does not match'}")

    if args.remove_pin:
        did_something = True
        db.remove_person_pin(args.remove_pin)
        print(f"[DB] PIN removed for id={args.remove_pin}")

    if not did_something:
        db.summary()

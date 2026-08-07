import argparse
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = "data/nova.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS emergency_contacts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL,
    phone    TEXT NOT NULL,
    role     TEXT DEFAULT '',
    priority INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS person_contacts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    name      TEXT NOT NULL,
    phone     TEXT NOT NULL,
    relation  TEXT DEFAULT ''
);
"""

class ContactsDB:
    def __init__(self, db_path: str = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        with self._conn() as c:
            c.executescript(SCHEMA)

    @staticmethod
    def _rows_to_dicts(rows, cols):
        return [dict(zip(cols, r)) for r in rows]

    def add_emergency_contact(self, name: str, phone: str,
                              role: str = "", priority: int = 1) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO emergency_contacts (name, phone, role, priority) "
                "VALUES (?, ?, ?, ?)", (name, phone, role, priority))
            return cur.lastrowid

    def list_emergency_contacts(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, name, phone, role, priority FROM emergency_contacts "
                "ORDER BY priority ASC, id ASC").fetchall()
        return self._rows_to_dicts(rows, ("id", "name", "phone", "role", "priority"))

    def remove_emergency_contact(self, contact_id: int):
        with self._conn() as c:
            c.execute("DELETE FROM emergency_contacts WHERE id=?", (contact_id,))

    def add_person_contact(self, person_id: int, name: str, phone: str,
                           relation: str = "") -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO person_contacts (person_id, name, phone, relation) "
                "VALUES (?, ?, ?, ?)", (person_id, name, phone, relation))
            return cur.lastrowid

    def list_person_contacts(self, person_id: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, person_id, name, phone, relation FROM person_contacts "
                "WHERE person_id=? ORDER BY id ASC", (person_id,)).fetchall()
        return self._rows_to_dicts(
            rows, ("id", "person_id", "name", "phone", "relation"))

    def remove_person_contact(self, contact_id: int):
        with self._conn() as c:
            c.execute("DELETE FROM person_contacts WHERE id=?", (contact_id,))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true",
                        help="Create the tables (safe to re-run)")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--list", action="store_true",
                        help="Print all emergency contacts")
    parser.add_argument("--add-emergency", nargs=2, metavar=("NAME", "PHONE"),
                        help='--add-emergency "Chikondi (Security)" +265998000000')
    parser.add_argument("--role", default="", help="Used with --add-emergency")
    parser.add_argument("--priority", type=int, default=1, help="Used with --add-emergency")
    parser.add_argument("--add-person-contact", nargs=3,
                        metavar=("PERSON_ID", "NAME", "PHONE"),
                        help='--add-person-contact 3 "Grace Wella" +265991111111')
    parser.add_argument("--relation", default="", help="Used with --add-person-contact")
    parser.add_argument("--list-person", type=int, metavar="PERSON_ID",
                        help="Print contacts for one specific person")
    args = parser.parse_args()

    db = ContactsDB(args.db)
    print(f"[ContactsDB] Ready at {args.db}")

    if args.add_emergency:
        name, phone = args.add_emergency
        cid = db.add_emergency_contact(name, phone, role=args.role, priority=args.priority)
        print(f"[ContactsDB] Added emergency contact id={cid}")

    if args.add_person_contact:
        pid, name, phone = args.add_person_contact
        cid = db.add_person_contact(int(pid), name, phone, relation=args.relation)
        print(f"[ContactsDB] Added person contact id={cid}")

    if args.list_person:
        for c in db.list_person_contacts(args.list_person):
            print(f"    [{c['id']}] {c['name']} ({c['relation']}) — {c['phone']}")

    if args.list or args.init or not any([args.add_emergency, args.add_person_contact, args.list_person]):
        ec = db.list_emergency_contacts()
        print(f"  {len(ec)} emergency contact(s):")
        for c in ec:
            print(f"    [{c['priority']}] {c['name']} ({c['role']}) — {c['phone']}")

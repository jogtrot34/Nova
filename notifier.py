"""
notifier.py

Calls and texts people through your GSM modem. Nothing here fires on
its own — every call in this file only runs when something else
explicitly tells it to (a voice command, a button in the web UI, or a
direct function call). That's on purpose, per the "leave it at
instructed to" scope.

Setup, once, wherever you start Nova (e.g. main.py):

    from emergency_dial import EmergencyDialer
    from contacts_db import ContactsDB
    from notifier import Notifier

    dialer   = EmergencyDialer()   # wraps modem.py's SimpleModem
    notifier = Notifier(dialer, ContactsDB())

Trigger it from wherever "instructed" lives for you — a recognised
voice command, a tool call from your brain/LLM layer, or (already
wired up) a button in the web UI:

    notifier.alert_emergency_contacts("Unknown person refused to leave.")

    # call the top emergency contact and play a prerecorded message
    # once they pick up, instead of just ringing and hanging up:
    notifier.alert_emergency_contacts(
        "Unknown person refused to leave.", also_call=True,
        play_audio_path="assets/voice/phrases/emergency_call_message.wav")

    notifier.notify_person_contacts(person_id=3,
        "Joseph could not be identified confidently and was denied entry.")
"""

from typing import Optional

from contacts_db import ContactsDB


class Notifier:
    def __init__(self, modem, contacts: Optional[ContactsDB] = None):
        """
        modem    : anything with .make_call(number) and
                   .send_sms(number, message). Pass an
                   emergency_dial.EmergencyDialer instead of a plain
                   modem to also get "play a message once answered".
        contacts : a ContactsDB instance (created for you if omitted)
        """
        self.modem    = modem
        self.contacts = contacts or ContactsDB()

    # ── low level ──────────────────────────────────────────────────────────

    def call(self, number: str) -> bool:
        result = self.modem.make_call(number)
        # SimpleModem returns a status string ("answered"/"denied"/
        # "busy"/"no_response"/"error"); older modem wrappers return a
        # plain bool. Handle both — a truthy non-empty string like
        # "denied" must NOT be read as a successful call.
        if isinstance(result, str):
            return result == "answered"
        return bool(result)

    def text(self, number: str, message: str) -> bool:
        return bool(self.modem.send_sms(number, message))

    def call_and_play(self, number: str, wav_path: str) -> dict:
        """Only works if self.modem is an EmergencyDialer (or anything
        else exposing a matching call_and_play method)."""
        if not hasattr(self.modem, "call_and_play"):
            return {"status": "error",
                    "error": "This modem doesn't support call_and_play — "
                             "pass an emergency_dial.EmergencyDialer to "
                             "Notifier() to enable it."}
        return self.modem.call_and_play(number, wav_path)

    # ── emergency contacts (global list, any person can trigger this) ──────

    def alert_emergency_contacts(self, message: str,
                                  also_call: bool = False,
                                  play_audio_path: Optional[str] = None) -> dict:
        """Texts every emergency contact. If also_call=True, additionally
        rings the single highest-priority contact (priority 1 = first).
        If play_audio_path is also given and self.modem supports it,
        plays that message into the call once it's answered instead of
        just ringing and hanging up."""
        results = {"texted": [], "called": [], "failed": []}
        contacts = self.contacts.list_emergency_contacts()

        for c in contacts:
            ok = self.text(c["phone"], f"[Nova Alert] {message}")
            (results["texted"] if ok else results["failed"]).append(c["name"])

        if also_call and contacts:
            target = contacts[0]   # lowest priority number = called first
            if play_audio_path and hasattr(self.modem, "call_and_play"):
                call_result = self.modem.call_and_play(target["phone"], play_audio_path)
                results["call_detail"] = call_result
                ok = call_result.get("status") == "answered"
            else:
                ok = self.call(target["phone"])
            (results["called"] if ok else results["failed"]).append(target["name"])

        return results

    # ── per-person contacts (e.g. next of kin for one enrolled person) ─────

    def notify_person_contacts(self, person_id: int, message: str,
                                also_call: bool = False,
                                play_audio_path: Optional[str] = None) -> dict:
        results = {"texted": [], "called": [], "failed": []}
        for c in self.contacts.list_person_contacts(person_id):
            ok = self.text(c["phone"], f"[Nova] {message}")
            (results["texted"] if ok else results["failed"]).append(c["name"])
            if also_call:
                if play_audio_path and hasattr(self.modem, "call_and_play"):
                    call_result = self.modem.call_and_play(c["phone"], play_audio_path)
                    results.setdefault("call_details", []).append(call_result)
                    ok = call_result.get("status") == "answered"
                else:
                    ok = self.call(c["phone"])
                (results["called"] if ok else results["failed"]).append(c["name"])
        return results


if __name__ == "__main__":
    import argparse

    from emergency_dial import EmergencyDialer
    from contacts_db import ContactsDB

    parser = argparse.ArgumentParser(description="Test notifier.py directly from the terminal")
    parser.add_argument("--port", default=None, help="Modem port, e.g. /dev/ttyACM0")
    parser.add_argument("--text", metavar="NUMBER", help="Send a one-off SMS")
    parser.add_argument("--message", default="Test message from Nova's notifier.py")
    parser.add_argument("--call", metavar="NUMBER", help="Make a one-off call")
    parser.add_argument("--emergency", action="store_true",
                        help="Run the full alert_emergency_contacts() flow "
                             "against whatever's in contacts_db.py")
    parser.add_argument("--also-call", action="store_true")
    parser.add_argument("--play", metavar="WAV_PATH",
                        help="Play this into the call once answered (needs --also-call)")
    args = parser.parse_args()

    dialer   = EmergencyDialer(port=args.port)
    notifier = Notifier(dialer, ContactsDB())

    if args.text:
        ok = notifier.text(args.text, args.message)
        print(f"SMS {'sent' if ok else 'failed'}")
    elif args.call:
        ok = notifier.call(args.call)
        print(f"Call {'answered' if ok else 'not answered / failed'}")
    elif args.emergency:
        result = notifier.alert_emergency_contacts(
            args.message, also_call=args.also_call, play_audio_path=args.play)
        print(result)
    else:
        parser.print_help()

from typing import Optional

from contacts_db import ContactsDB

class Notifier:
    def __init__(self, modem, contacts: Optional[ContactsDB] = None):
        self.modem = modem
        self.contacts = contacts or ContactsDB()

    def call(self, number: str) -> bool:
        result = self.modem.make_call(number)
        if isinstance(result, str):
            return result == "answered"
        return bool(result)

    def text(self, number: str, message: str) -> bool:
        return bool(self.modem.send_sms(number, message))

    def call_and_play(self, number: str, wav_path: str) -> dict:
        if not hasattr(self.modem, "call_and_play"):
            return {"status": "error",
                    "error": "This modem doesn't support call_and_play — "
                             "pass an emergency_dial.EmergencyDialer to "
                             "Notifier() to enable it."}
        return self.modem.call_and_play(number, wav_path)

    def alert_emergency_contacts(self, message: str,
                                 also_call: bool = False,
                                 play_audio_path: Optional[str] = None) -> dict:
        results = {"texted": [], "called": [], "failed": []}
        contacts = self.contacts.list_emergency_contacts()

        for c in contacts:
            ok = self.text(c["phone"], f"[Nova Alert] {message}")
            (results["texted"] if ok else results["failed"]).append(c["name"])

        if also_call and contacts:
            target = contacts[0]
            if play_audio_path and hasattr(self.modem, "call_and_play"):
                call_result = self.modem.call_and_play(target["phone"], play_audio_path)
                results["call_detail"] = call_result
                ok = call_result.get("status") == "answered"
            else:
                ok = self.call(target["phone"])
            (results["called"] if ok else results["failed"]).append(target["name"])

        return results

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

    dialer = EmergencyDialer(port=args.port)
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

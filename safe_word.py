"""
safe_word.py

A duress/panic phrase — say it once anywhere in earshot and Nova
starts watching for it again; say it a second time within 30 seconds
and it texts your #1 priority emergency contact "something might be
wrong." One mention alone does nothing except arm the window — that's
what makes it safe to actually use without constant false alarms, and
hard to trigger by accident (you'd need to say it twice, on purpose,
inside 30 seconds).

Runs as its own background thread, completely independent of the
wake-word/camera pipeline — it's listening continuously, not just
after "Nova". Needs internet (uses the same Google speech-recognition
backend fast_responder.py does — swap for an offline STT later if you
want this offline too; noted as a known limitation, not fixed here
given the deadline).

Configure the phrase via environment variable so it's never sitting in
a script anyone can read:

    NOVA_SAFE_WORD="pineapple" python3 main.py

Wired into main.py. Test it standalone first:

    python3 safe_word.py --test          # types instead of listening
    NOVA_SAFE_WORD=pineapple python3 safe_word.py --listen
"""

import os
import threading
import time
from typing import Optional

SAFE_WORD = os.environ.get("NOVA_SAFE_WORD", "pineapple").strip().lower()
REPEAT_WINDOW_SECONDS = 30
LISTEN_CLIP_SECONDS = 4


class SafeWordListener:
    def __init__(self, notifier=None, ui=None, contacts_db=None):
        self.notifier = notifier   # optional notifier.Notifier
        self.ui = ui                # optional NovaWebUI — for orange orb + event log
        self._contacts_db = contacts_db
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._armed_at: Optional[float] = None
        self._lock = threading.Lock()

    def set_notifier(self, notifier):
        self.notifier = notifier

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[SafeWord] Listening ({LISTEN_CLIP_SECONDS}s clips, "
              f"{REPEAT_WINDOW_SECONDS}s repeat window).")

    def stop(self):
        self._running = False

    # ── internals ────────────────────────────────────────────────────────

    def _heard(self, text: str) -> bool:
        return SAFE_WORD in text.lower()

    def _listen_once(self) -> Optional[str]:
        try:
            import speech_recognition as sr
        except Exception as e:
            print(f"[SafeWord] speech_recognition not available: {e}")
            time.sleep(5)   # don't spin-loop if the dependency is missing
            return None
        try:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=LISTEN_CLIP_SECONDS,
                                          phrase_time_limit=LISTEN_CLIP_SECONDS)
            return recognizer.recognize_google(audio)
        except Exception:
            # timeouts, "couldn't understand", no internet, mic busy — all
            # just mean "didn't hear the safe word this round," not a
            # crash-worthy problem.
            return None

    def _trigger_alert(self):
        from contacts_db import ContactsDB
        db = self._contacts_db or ContactsDB()
        contacts = db.list_emergency_contacts()

        if not contacts:
            print("[SafeWord] Triggered, but no emergency contacts configured.")
            if self.ui:
                self.ui._log_event("info", "Safe word triggered \u2014 no "
                                   "emergency contacts configured", "")
            return

        target   = contacts[0]   # "emergency contact 1" = top priority
        message  = ("Something might be wrong. This is an automated "
                   "safe-word alert from Nova.")
        # Prefer whatever notifier is live right now — the modem may
        # have been connected after startup via the Controls panel,
        # not necessarily passed in when this listener was created.
        notifier = self.notifier or (getattr(self.ui, "_notifier", None) if self.ui else None)

        if notifier:
            ok = notifier.text(target["phone"], message)
            print(f"[SafeWord] Alert {'sent' if ok else 'FAILED'} to {target['name']}")
        else:
            print(f"[SafeWord] Triggered, but no modem connected \u2014 would "
                  f"have texted {target['name']} ({target['phone']}): {message}")

        if self.ui:
            self.ui._log_event("info", "SAFE WORD TRIGGERED",
                               f"Alerted {target['name']}")

    def _run(self):
        while self._running:
            text = self._listen_once()
            if not text or not self._heard(text):
                continue

            with self._lock:
                now = time.time()
                repeated = (self._armed_at is not None and
                           (now - self._armed_at) <= REPEAT_WINDOW_SECONDS)

                if repeated:
                    print("[SafeWord] Second mention within window \u2014 triggering alert.")
                    self._armed_at = None
                    if self.ui:
                        self.ui.set_status("Monitoring...", "idle")
                    self._trigger_alert()
                else:
                    print(f"[SafeWord] Heard once \u2014 armed, waiting for a "
                          f"repeat within {REPEAT_WINDOW_SECONDS}s.")
                    self._armed_at = now
                    if self.ui:
                        self.ui.set_status(
                            "Safe word heard \u2014 say it again to alert",
                            "safe_word")
                    threading.Thread(target=self._expire, args=(now,),
                                     daemon=True).start()

    def _expire(self, armed_ts: float):
        time.sleep(REPEAT_WINDOW_SECONDS)
        with self._lock:
            if self._armed_at == armed_ts:
                self._armed_at = None
                if self.ui:
                    self.ui.set_status("Monitoring...", "idle")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", action="store_true",
                        help="Actually listen on the mic")
    parser.add_argument("--test", action="store_true",
                        help="Type the phrase instead of speaking it "
                             "(no mic/internet needed)")
    args = parser.parse_args()

    print(f"Safe word is: {SAFE_WORD!r} "
          f"(set NOVA_SAFE_WORD to change it)")

    if args.test:
        listener = SafeWordListener()
        while True:
            text = input("Type what was 'said' (empty to quit): ").strip()
            if not text:
                break
            if not listener._heard(text):
                print("  (not heard)")
                continue
            with listener._lock:
                now = time.time()
                repeated = (listener._armed_at is not None and
                           (now - listener._armed_at) <= REPEAT_WINDOW_SECONDS)
                if repeated:
                    print("  -> SECOND MENTION -> would trigger alert now")
                    listener._armed_at = None
                else:
                    print(f"  -> armed, say it again within {REPEAT_WINDOW_SECONDS}s")
                    listener._armed_at = now
    elif args.listen:
        listener = SafeWordListener()
        listener.start()
        print("Listening... Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            listener.stop()
    else:
        parser.print_help()

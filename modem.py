import serial
import serial.tools.list_ports
import time
import threading
import re

class SimpleModem:
    def __init__(self):
        self.modem = None
        self.port = None
        self.baud = 115200
        self.in_call = False
        self.call_answered = False
        self.call_completed = threading.Event()
        self.call_result = None
        self.monitoring = False
        self.monitor_thread = None

    def connect(self, port=None):
        if port is None:
            known_ports = ["/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyACM2"]
            for p in known_ports:
                try:
                    import os
                    if os.path.exists(p):
                        test_ser = serial.Serial(p, self.baud, timeout=1)
                        test_ser.write(b'AT\r\n')
                        time.sleep(0.5)
                        response = test_ser.read(50)
                        if b'OK' in response:
                            port = p
                            test_ser.close()
                            print(f"✅ Found working port: {p}")
                            break
                        test_ser.close()
                except:
                    continue

        if port is None:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                if 'ACM' in p.device or 'USB' in p.device:
                    port = p.device
                    break

        if port is None:
            print("❌ No modem found")
            return False

        try:
            self.modem = serial.Serial(port, self.baud, timeout=2)
            time.sleep(2)
            self.port = port

            self.send_at("AT")
            self.send_at("AT+CMGF=1")
            self.send_at("AT+CLIP=0")
            self.send_at("AT+COLP=1")
            self.send_at("AT+CRC=0")

            print(f"✅ Connected to modem on {port}")
            return True

        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False

    def send_at(self, cmd, timeout=5):
        if not self.modem:
            return None

        try:
            with threading.Lock():
                self.modem.reset_input_buffer()
                self.modem.write((cmd + "\r").encode())
                time.sleep(0.2)

                deadline = time.time() + timeout
                response = ""
                while time.time() < deadline:
                    if self.modem.in_waiting:
                        response += self.modem.read(self.modem.in_waiting).decode(errors="ignore")
                        if "OK" in response or "ERROR" in response or ">" in response:
                            break
                    time.sleep(0.05)

                return response.strip()
        except Exception as e:
            return f"ERROR: {e}"

    def make_call(self, number, timeout=30, hangup_on_answer=True):
        number = re.sub(r'[^0-9+]', '', number)

        print(f"\n📞 Calling {number}...")

        self.call_completed.clear()
        self.call_result = None
        self.call_answered = False
        self.in_call = True

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_outgoing_call, daemon=True)
        self.monitor_thread.start()

        response = self.send_at(f"ATD{number};", timeout=5)

        if response and "ERROR" in response:
            self.in_call = False
            self.monitoring = False
            print(f"❌ Call failed: {response}")
            return "error"

        print(f"📞 Ringing... (waiting up to {timeout}s)")

        if self.call_completed.wait(timeout):
            result = self.call_result
            if result == "answered":
                print("✅ Call was ANSWERED!")
                time.sleep(1)
                if hangup_on_answer:
                    self.hangup()
            elif result == "denied":
                print("❌ Call was DENIED/REJECTED")
            elif result == "busy":
                print("❌ Number was BUSY")
            else:
                print("⚠️  Call ended with unknown status")
            return result
        else:
            self.monitoring = False
            if self.in_call:
                self.hangup()
            self.call_result = "no_response"
            print("⏰ No answer (timed out)")
            return "no_response"

    def _monitor_outgoing_call(self):
        buffer = ""

        while self.monitoring and self.in_call:
            if self.modem and self.modem.in_waiting:
                try:
                    data = self.modem.read(self.modem.in_waiting).decode(errors="ignore")
                    buffer += data

                    lines = buffer.split("\r\n")
                    buffer = lines[-1] if lines else ""

                    for line in lines[:-1]:
                        line = line.strip()

                        if ("+COLP:" in line or
                            "CONNECT" in line or
                            "VOICE" in line and "CONNECT" in line or
                            "ANSWER" in line):
                            self.call_answered = True
                            self.call_result = "answered"
                            print("📞 Call connected! (answered)")
                            self.in_call = False
                            self.monitoring = False
                            self.call_completed.set()
                            return

                        if "NO CARRIER" in line:
                            if self.call_answered:
                                self.call_result = "answered"
                            else:
                                if "RINGBACK" in buffer:
                                    self.call_result = "denied"
                                else:
                                    self.call_result = "denied"

                            self.in_call = False
                            self.monitoring = False
                            self.call_completed.set()
                            return

                        if "BUSY" in line:
                            self.call_result = "busy"
                            self.in_call = False
                            self.monitoring = False
                            self.call_completed.set()
                            return

                        if "NO ANSWER" in line:
                            self.call_result = "no_response"
                            self.in_call = False
                            self.monitoring = False
                            self.call_completed.set()
                            return

                        if "VOICE NO CARRIER" in line:
                            if self.call_answered:
                                self.call_result = "answered"
                            else:
                                if "RINGBACK" in buffer:
                                    self.call_result = "denied"
                                else:
                                    self.call_result = "no_response"

                            self.in_call = False
                            self.monitoring = False
                            self.call_completed.set()
                            return

                        if "RINGBACK" in line:
                            print("📞 Ringing... (ringback detected)")

                        if "RING" in line and "RINGBACK" not in line:
                            pass

                except Exception as e:
                    print(f"Monitor error: {e}")

            time.sleep(0.1)

        if self.in_call:
            self.in_call = False
            if self.call_result is None:
                self.call_result = "no_response"
            self.call_completed.set()

    def hangup(self):
        print("📞 Attempting to hang up...")

        methods = [
            ("ATH", 2),
            ("ATH0", 2),
            ("+++", 1),
            ("ATH", 2),
        ]

        for cmd, wait in methods:
            try:
                if cmd == "+++":
                    self.modem.write(b'+++')
                    time.sleep(1)
                    self.modem.write(b'ATH\r')
                else:
                    response = self.send_at(cmd, timeout=wait)
                    print(f"  Sent {cmd}: {response}")

                time.sleep(0.5)
                if not self.in_call:
                    break

            except Exception as e:
                print(f"  Error with {cmd}: {e}")

        if self.in_call:
            print("⚠️  Force resetting modem...")
            try:
                self.send_at("AT+CFUN=1,1", timeout=5)
                time.sleep(2)
            except:
                pass

        self.in_call = False
        self.monitoring = False
        print("📞 Call ended")

    def send_sms(self, number, message):
        number = re.sub(r'[^0-9+]', '', number)

        print(f"\n📩 Sending SMS to {number}...")

        try:
            self.send_at("AT+CMGF=1", timeout=2)

            response = self.send_at(f'AT+CMGS="{number}"', timeout=3)

            if response and ">" in response:
                self.modem.write((message + chr(26)).encode())
                time.sleep(3)

                response = ""
                deadline = time.time() + 10
                while time.time() < deadline:
                    if self.modem.in_waiting:
                        response += self.modem.read(self.modem.in_waiting).decode(errors="ignore")
                        if "OK" in response or "ERROR" in response:
                            break
                    time.sleep(0.1)

                if "OK" in response or "+CMGS" in response:
                    print("✅ SMS sent successfully!")
                    return True
                else:
                    print(f"❌ SMS send failed: {response}")
                    return False
            else:
                print(f"❌ Failed to get '>' prompt: {response}")
                return False

        except Exception as e:
            print(f"❌ Error sending SMS: {e}")
            return False

    def disconnect(self):
        self.monitoring = False
        if self.in_call:
            self.hangup()

        if self.modem:
            self.modem.close()
            print("Disconnected from modem")

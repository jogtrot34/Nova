import flet as ft
import threading
import time
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    _SD_AVAILABLE = False

BG = "#0a0a0f"
BG_CARD = "#111118"
BG_CARD2 = "#16161f"
GREEN = "#4ade80"
GREEN_DIM = "#1a3d26"
ORANGE = "#fb923c"
YELLOW = "#facc15"
BLUE = "#60a5fa"
AMBER = "#f59e0b"
RED = "#f87171"
TEXT = "#e8e8f0"
TEXT_DIM = "#6b7280"
TEXT_FAINT = "#374151"
BORDER = "#1f2937"

def _p(h=0, v=0):
    return ft.Padding(left=h, top=v, right=h, bottom=v)

def _m(h=0, v=0):
    return ft.Margin(left=h, top=v, right=h, bottom=v)

class OrbState:
    IDLE = "idle"
    DETECTED = "detected"
    IDENTIFYING = "identifying"
    SPEAKING = "speaking"
    CONFIRMED = "confirmed"
    WAITING = "waiting"
    DENIED = "denied"

ORB_COLORS = {
    OrbState.IDLE: GREEN,
    OrbState.DETECTED: ORANGE,
    OrbState.IDENTIFYING: YELLOW,
    OrbState.SPEAKING: BLUE,
    OrbState.CONFIRMED: GREEN,
    OrbState.WAITING: AMBER,
    OrbState.DENIED: RED,
}

ORB_SPEED = {
    OrbState.IDLE: 2.5,
    OrbState.DETECTED: 0.6,
    OrbState.IDENTIFYING: 0.3,
    OrbState.SPEAKING: 0.15,
    OrbState.CONFIRMED: 0.4,
    OrbState.WAITING: 0.8,
    OrbState.DENIED: 0.25,
}

@dataclass
class Verdict:
    track_id: int
    name: str = "Unknown"
    person_id: Optional[int] = None
    face_conf: float = 0.0
    voice_conf: float = 0.0
    combined_conf: float = 0.0
    method: str = ""
    access: str = "none"
    decision: str = "denied"
    top_desc: str = "not visible"
    bottom_desc: str = "not visible"
    gender: str = "unknown"
    conflict: bool = False
    needs_voice: bool = False

class MicMonitor:
    def __init__(self, device: int = 1, samplerate: int = 44100,
                 blocksize: int = 512):
        self._amplitude = 0.0
        self._running = False
        self._lock = threading.Lock()
        self._device = device
        self._samplerate = samplerate
        self._blocksize = blocksize

    def _callback(self, indata, frames, time_info, status):
        rms = float(np.sqrt(np.mean(indata ** 2)))
        normalised = min(rms * 12.0, 1.0)
        with self._lock:
            self._amplitude = self._amplitude * 0.7 + normalised * 0.3

    def start(self):
        if not _SD_AVAILABLE:
            return
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            with sd.InputStream(
                device=self._device,
                channels=1,
                samplerate=self._samplerate,
                blocksize=self._blocksize,
                callback=self._callback,
                dtype="float32",
            ):
                while self._running:
                    time.sleep(0.05)
        except Exception as e:
            print(f"[MicMonitor] Could not open mic: {e}")

    def stop(self):
        self._running = False

    @property
    def amplitude(self) -> float:
        with self._lock:
            return self._amplitude

_ORB_SIZE = 180
_CORE_SIZE = 52
_RING_SIZES = [180, 140, 100, 68]

def build_orb(state: str, tick: float,
              amplitude: float = 0.0) -> ft.Stack:
    color = ORB_COLORS.get(state, GREEN)
    speed = ORB_SPEED.get(state, 2.0)
    phase = (tick % speed) / speed

    wave_boost = amplitude * 0.6

    ring_configs = [
        (_ORB_SIZE, 0.06, 0.00, 0.08 + wave_boost * 0.5),
        (_ORB_SIZE - 40, 0.10, 0.15, 0.10 + wave_boost * 0.7),
        (_ORB_SIZE - 80, 0.16, 0.30, 0.12 + wave_boost * 0.9),
        (_ORB_SIZE - 112, 0.22, 0.45, 0.15 + wave_boost * 1.2),
    ]

    half = _ORB_SIZE // 2
    controls = []

    for size, base_op, offset, react in ring_configs:
        p = ((phase + offset) % 1.0)
        pulse = abs(math.sin(p * math.pi))

        opacity = base_op + pulse * 0.35
        scale_v = 1.0 + pulse * react

        ring_half = size // 2
        margin = half - ring_half

        controls.append(ft.Container(
            width=size, height=size,
            border_radius=ring_half,
            border=ft.Border(
                top=ft.BorderSide(1.0, color),
                bottom=ft.BorderSide(1.0, color),
                left=ft.BorderSide(1.0, color),
                right=ft.BorderSide(1.0, color),
            ),
            opacity=opacity,
            scale=ft.Scale(scale=scale_v),
            animate_scale=ft.Animation(80, ft.AnimationCurve.EASE_OUT),
            animate_opacity=ft.Animation(80, ft.AnimationCurve.EASE_OUT),
            left=margin, top=margin,
        ))

    core_pulse = 0.70 + abs(math.sin(phase * math.pi * 2)) * 0.30
    core_scale = 1.0 + wave_boost * 0.4 + abs(math.sin(phase * math.pi)) * 0.15
    core_margin = half - _CORE_SIZE // 2

    core = ft.Container(
        width=_CORE_SIZE, height=_CORE_SIZE,
        border_radius=_CORE_SIZE // 2,
        gradient=ft.RadialGradient(
            colors=[color + "ff", color + "88", color + "00"],
            radius=0.6,
        ),
        opacity=core_pulse,
        scale=ft.Scale(scale=core_scale),
        animate_scale=ft.Animation(60, ft.AnimationCurve.EASE_OUT),
        animate_opacity=ft.Animation(60, ft.AnimationCurve.EASE_OUT),
        left=core_margin, top=core_margin,
        shadow=ft.BoxShadow(
            blur_radius=32,
            color=color + "66",
            offset=ft.Offset(0, 0),
        ),
    )
    controls.append(core)

    if amplitude > 0.05:
        n_bars = 8
        for i in range(n_bars):
            angle = (i / n_bars) * 2 * math.pi
            bar_amp = amplitude * (0.6 + 0.4 * abs(math.sin(
                angle + tick * 8)))
            bar_h = int(4 + bar_amp * 28)
            dist = _CORE_SIZE // 2 + 6
            bx = int(half + dist * math.cos(angle) - 2)
            by = int(half + dist * math.sin(angle) - bar_h // 2)
            controls.append(ft.Container(
                width=4, height=bar_h,
                border_radius=2,
                bgcolor=color,
                opacity=0.5 + bar_amp * 0.5,
                left=bx, top=by,
            ))

    return ft.Stack(
        width=_ORB_SIZE,
        height=_ORB_SIZE,
        controls=controls,
    )

def build_track_card(v: Verdict) -> ft.Container:
    is_known = v.name != "Unknown" and not v.conflict
    accent = RED if (not is_known or v.conflict) else GREEN
    status = ("⚠ CONFLICT" if v.conflict
              else "AUTHORIZED" if is_known
              else "UNKNOWN")
    dec_color = GREEN if v.decision == "granted" else RED

    def bar_row(label, value, color):
        filled = max(4.0, 280.0 * float(value))
        return ft.Column(spacing=3, controls=[
            ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    ft.Text(label, size=10, color=TEXT_DIM,
                            weight=ft.FontWeight.W_500),
                    ft.Text(f"{value:.0%}", size=10, color=color,
                            weight=ft.FontWeight.W_600),
                ]
            ),
            ft.Container(
                height=3, border_radius=99, bgcolor=BG_CARD2,
                clip_behavior=ft.ClipBehavior.HARD_EDGE,
                content=ft.Container(
                    height=3, width=filled, border_radius=99,
                    bgcolor=color,
                    animate_size=ft.Animation(500, ft.AnimationCurve.EASE_OUT),
                )
            ),
        ])

    items = [
        ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.Row(spacing=6, controls=[
                    ft.Container(width=6, height=6,
                                 border_radius=3, bgcolor=accent),
                    ft.Text(f"Track {v.track_id:02d}", size=10,
                            color=TEXT_DIM, weight=ft.FontWeight.W_500),
                ]),
                ft.Text(status, size=10, color=dec_color,
                        weight=ft.FontWeight.W_700),
            ]
        ),
        ft.Text(v.name, size=18, weight=ft.FontWeight.W_700, color=TEXT),
        ft.Text(f"{v.top_desc}  ·  {v.bottom_desc}",
                size=11, color=TEXT_DIM),
    ]

    if v.needs_voice:
        items.append(ft.Container(
            padding=_p(8, 3), border_radius=99, bgcolor="#2d2000",
            border=ft.Border(
                top=ft.BorderSide(0.5, AMBER),
                bottom=ft.BorderSide(0.5, AMBER),
                left=ft.BorderSide(0.5, AMBER),
                right=ft.BorderSide(0.5, AMBER),
            ),
            content=ft.Text("🎤  Say something to verify",
                            size=10, color=AMBER),
        ))

    items += [
        ft.Divider(height=1, color=BORDER),
        bar_row("Face", v.face_conf, GREEN),
        bar_row("Voice", v.voice_conf, BLUE),
        bar_row("Combined", v.combined_conf, dec_color),
        ft.Text(f"Method: {v.method or '—'}",
                size=10, color=TEXT_FAINT),
    ]

    return ft.Container(
        margin=_m(v=4),
        padding=_p(14, 14),
        border_radius=12,
        bgcolor=BG_CARD,
        border=ft.Border(
            top=ft.BorderSide(0.5, accent),
            bottom=ft.BorderSide(0.5, BORDER),
            left=ft.BorderSide(2.0, accent),
            right=ft.BorderSide(0.5, BORDER),
        ),
        content=ft.Column(spacing=8, controls=items),
    )

class NovaUI:
    def __init__(self, mic_device: int = 1):
        self._verdicts: list[Verdict] = []
        self._status: str = "Monitoring..."
        self._orb_state: str = OrbState.IDLE
        self._tick: float = 0.0
        self._lock = threading.Lock()
        self._page: Optional[ft.Page] = None
        self._orb_wrap: Optional[ft.Container] = None
        self._status_txt: Optional[ft.Text] = None
        self._tracks_col: Optional[ft.Column] = None
        self._mic = MicMonitor(device=mic_device)

    def push_verdict(self, verdict: Verdict):
        with self._lock:
            self._verdicts = [v for v in self._verdicts
                              if v.track_id != verdict.track_id]
            self._verdicts.append(verdict)
            self._verdicts.sort(key=lambda v: v.track_id)
            if verdict.conflict:
                self._orb_state = OrbState.DENIED
            elif verdict.needs_voice:
                self._orb_state = OrbState.WAITING
            elif verdict.decision == "granted":
                self._orb_state = OrbState.CONFIRMED
            else:
                self._orb_state = OrbState.DENIED
        self._refresh_tracks()

    def set_detected(self):
        self._orb_state = OrbState.DETECTED

    def set_identifying(self):
        self._orb_state = OrbState.IDENTIFYING

    def set_speaking(self, speaking: bool):
        self._orb_state = (OrbState.SPEAKING if speaking
                           else OrbState.IDLE)

    def remove_track(self, track_id: int):
        with self._lock:
            self._verdicts = [v for v in self._verdicts
                              if v.track_id != track_id]
            if not self._verdicts:
                self._orb_state = OrbState.IDLE
        self._refresh_tracks()

    def clear_tracks(self):
        with self._lock:
            self._verdicts.clear()
            self._orb_state = OrbState.IDLE
        self._refresh_tracks()

    def set_status(self, text: str,
                   orb_state: Optional[str] = None):
        self._status = text
        if orb_state:
            self._orb_state = orb_state
        if self._status_txt and self._page:
            self._status_txt.value = text
            try:
                self._page.update()
            except Exception:
                pass

    def run(self):
        ft.app(target=self._build)

    def _build(self, page: ft.Page):
        self._page = page
        page.title = "Nova Security Intelligence"
        page.bgcolor = BG
        page.padding = ft.Padding(left=0, top=0, right=0, bottom=0)
        page.window = ft.Window(width=440, height=860)

        header = ft.Container(
            padding=_p(20, 14),
            border=ft.Border(bottom=ft.BorderSide(0.5, BORDER)),
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    ft.Row(spacing=8, controls=[
                        ft.Container(
                            width=8, height=8,
                            border_radius=4, bgcolor=GREEN,
                            animate_opacity=ft.Animation(
                                1200, ft.AnimationCurve.EASE_IN_OUT),
                        ),
                        ft.Text("NOVA", size=15,
                                weight=ft.FontWeight.W_700,
                                color=TEXT),
                        ft.Text("Security Intelligence",
                                size=11, color=TEXT_DIM),
                    ]),
                    ft.Container(
                        padding=_p(10, 4), border_radius=99,
                        border=ft.Border(
                            top=ft.BorderSide(0.5, GREEN_DIM),
                            bottom=ft.BorderSide(0.5, GREEN_DIM),
                            left=ft.BorderSide(0.5, GREEN_DIM),
                            right=ft.BorderSide(0.5, GREEN_DIM),
                        ),
                        content=ft.Text("Active", size=10,
                                        color=GREEN,
                                        weight=ft.FontWeight.W_500),
                    ),
                ]
            )
        )

        self._orb_wrap = ft.Container(
            width=_ORB_SIZE, height=_ORB_SIZE,
            content=build_orb(self._orb_state, 0.0, 0.0),
            alignment=ft.Alignment(0, 0),
        )
        self._status_txt = ft.Text(
            self._status, size=13, color=TEXT_DIM,
            italic=True, text_align=ft.TextAlign.CENTER,
        )
        orb_section = ft.Container(
            padding=_p(0, 32),
            content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=16,
                controls=[self._orb_wrap, self._status_txt],
            ),
        )

        self._tracks_col = ft.Column(
            spacing=0,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[self._empty_state()],
        )

        bottom_bar = ft.Container(
            padding=_p(16, 10),
            border=ft.Border(top=ft.BorderSide(0.5, BORDER)),
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_AROUND,
                controls=[
                    ft.IconButton(
                        icon=ft.Icons.VIDEOCAM_OFF,
                        icon_color=TEXT_DIM, icon_size=22,
                        tooltip="Toggle camera",
                        on_click=lambda e: None,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.LIST_ALT,
                        icon_color=TEXT_DIM, icon_size=22,
                        tooltip="Access log",
                        on_click=self._show_log,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.PERSON_ADD,
                        icon_color=TEXT_DIM, icon_size=22,
                        tooltip="Enroll person",
                        on_click=self._show_enroll,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.REFRESH,
                        icon_color=TEXT_DIM, icon_size=22,
                        tooltip="Clear tracks",
                        on_click=lambda e: self.clear_tracks(),
                    ),
                ],
            )
        )

        page.add(ft.Column(
            expand=True, spacing=0,
            controls=[
                header,
                orb_section,
                ft.Divider(height=0.5, color=BORDER),
                ft.Container(
                    expand=True,
                    padding=_p(16, 0),
                    content=self._tracks_col,
                ),
                bottom_bar,
            ],
        ))

        self._mic.start()
        threading.Thread(target=self._orb_loop, daemon=True).start()

    def _orb_loop(self):
        start = time.time()
        while True:
            time.sleep(0.05)
            self._tick = time.time() - start
            amp = self._mic.amplitude
            if self._page and self._orb_wrap:
                try:
                    self._orb_wrap.content = build_orb(
                        self._orb_state, self._tick, amp)
                    self._page.update()
                except Exception:
                    pass

    def _empty_state(self):
        return ft.Container(
            padding=_p(0, 24),
            alignment=ft.Alignment(0, 0),
            content=ft.Text("No one detected",
                            size=12, color=TEXT_FAINT,
                            text_align=ft.TextAlign.CENTER),
        )

    def _refresh_tracks(self):
        if not self._tracks_col or not self._page:
            return
        with self._lock:
            verdicts = list(self._verdicts)
        self._tracks_col.controls = (
            [build_track_card(v) for v in verdicts]
            if verdicts else [self._empty_state()]
        )
        try:
            self._page.update()
        except Exception:
            pass

    def _show_log(self, e):
        try:
            from db import NovaDB
            logs = NovaDB().get_access_log(20)
        except Exception:
            logs = []

        rows = []
        for log in logs:
            name = f"{log.get('first_name','')}" \
                   f" {log.get('last_name','')}".strip() or "Unknown"
            color = GREEN if log["decision"] == "granted" else RED
            rows.append(ft.Container(
                padding=_p(4, 6),
                border=ft.Border(bottom=ft.BorderSide(0.5, BORDER)),
                content=ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Column(spacing=2, controls=[
                            ft.Text(name, size=12, color=TEXT,
                                    weight=ft.FontWeight.W_500),
                            ft.Text(str(log["timestamp"])[:19],
                                    size=10, color=TEXT_DIM),
                        ]),
                        ft.Column(
                            spacing=2,
                            horizontal_alignment=ft.CrossAxisAlignment.END,
                            controls=[
                                ft.Text(str(log["decision"]).upper(),
                                        size=10, color=color,
                                        weight=ft.FontWeight.W_700),
                                ft.Text(f"{log['confidence']:.0%}",
                                        size=10, color=TEXT_DIM),
                            ]
                        ),
                    ]
                )
            ))

        dlg = ft.AlertDialog(
            bgcolor=BG_CARD,
            title=ft.Text("Access Log", color=TEXT,
                          weight=ft.FontWeight.W_600),
            content=ft.Container(
                width=320, height=380,
                content=ft.Column(
                    scroll=ft.ScrollMode.AUTO,
                    controls=rows or [
                        ft.Text("No entries yet.",
                                color=TEXT_DIM, size=12)],
                ),
            ),
            actions=[ft.TextButton(
                "Close", on_click=lambda e: self._close(dlg))],
        )
        self._page.overlay.append(dlg)
        dlg.open = True
        self._page.update()

    def _show_enroll(self, e):
        first = ft.TextField(
            label="First name", bgcolor=BG_CARD2, color=TEXT,
            label_style=ft.TextStyle(color=TEXT_DIM),
            border_color=BORDER, focused_border_color=GREEN,
        )
        last = ft.TextField(
            label="Last name", bgcolor=BG_CARD2, color=TEXT,
            label_style=ft.TextStyle(color=TEXT_DIM),
            border_color=BORDER, focused_border_color=GREEN,
        )
        role = ft.TextField(
            label="Role  (owner / staff / visitor)",
            bgcolor=BG_CARD2, color=TEXT,
            label_style=ft.TextStyle(color=TEXT_DIM),
            border_color=BORDER, focused_border_color=GREEN,
        )
        msg = ft.Text("", size=11)

        def do_enroll(e):
            if not first.value or not last.value:
                msg.value = "First and last name required."
                msg.color = RED
                self._page.update()
                return
            try:
                from db import NovaDB
                pid = NovaDB().add_person(
                    first.value, last.value,
                    role=role.value or "visitor",
                    access_level="full" if role.value == "owner"
                    else "limited",
                )
                msg.value = (f"Registered (id={pid}). "
                             f"Run face + voice enroll next.")
                msg.color = GREEN
            except Exception as ex:
                msg.value = str(ex)
                msg.color = RED
            self._page.update()

        dlg = ft.AlertDialog(
            bgcolor=BG_CARD,
            title=ft.Text("Enroll New Person", color=TEXT,
                          weight=ft.FontWeight.W_600),
            content=ft.Container(
                width=300,
                content=ft.Column(spacing=10,
                                  controls=[first, last, role, msg]),
            ),
            actions=[
                ft.TextButton("Cancel",
                              on_click=lambda e: self._close(dlg)),
                ft.TextButton("Register", on_click=do_enroll,
                              style=ft.ButtonStyle(color=GREEN)),
            ],
        )
        self._page.overlay.append(dlg)
        dlg.open = True
        self._page.update()

    def _close(self, dlg):
        dlg.open = False
        self._page.update()

def _demo():
    ui = NovaUI(mic_device=1)

    def feed():
        time.sleep(1.5)
        ui.set_status("Motion detected...", OrbState.DETECTED)
        time.sleep(1.5)
        ui.set_status("Identifying...", OrbState.IDENTIFYING)
        time.sleep(2)
        ui.push_verdict(Verdict(
            track_id=1, name="Joseph Wella", person_id=1,
            face_conf=0.55, voice_conf=0.92, combined_conf=0.78,
            method="face + voice", access="full", decision="granted",
            top_desc="white Madrid jersey", bottom_desc="grey shorts",
            gender="male",
        ))
        ui.set_status("Joseph confirmed — access granted",
                      OrbState.CONFIRMED)
        time.sleep(4)

        ui.set_status("Nova speaking...", OrbState.SPEAKING)
        time.sleep(3)
        ui.set_status("Monitoring...", OrbState.IDLE)
        time.sleep(3)

        ui.push_verdict(Verdict(
            track_id=2, name="Unknown", person_id=None,
            face_conf=0.18, voice_conf=0.0, combined_conf=0.0,
            method="face", access="none", decision="denied",
            top_desc="blue shirt", bottom_desc="dark jeans",
            gender="male", needs_voice=True,
        ))
        ui.set_status("Unknown person — voice verification required",
                      OrbState.WAITING)
        time.sleep(4)

        ui.set_status("Voice did not match — access denied",
                      OrbState.DENIED)
        time.sleep(3)
        ui.remove_track(2)
        ui.set_status("Monitoring...", OrbState.IDLE)

    threading.Thread(target=feed, daemon=True).start()
    ui.run()

if __name__ == "__main__":
    _demo()

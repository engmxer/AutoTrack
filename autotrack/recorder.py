"""Запись действий физического геймпада.

Читаем физический джойстик через ``pygame`` (SDL2) с фиксированной частотой
(по умолчанию 250 Гц) по тому же точному планировщику, что и воспроизведение —
поэтому запись получается ровной по времени, а «дыры» (если система подвисла)
видны в отчёте.

Возможности:

* автоподбор устройства и разбор SDL-маппинга (стики, курки, кнопки, крестовина);
* автоопределение «-1..1» или «0..1» у курков;
* метки на трассе кнопкой или клавишей ``m``;
* аварийная остановка (``q``/``Esc``, кнопка-стоп, комбинация кнопок);
* окно с живым отображением стиков (по желанию);
* пост-обработка сразу после записи (сглаживание, ровная сетка).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .frame import BUTTON_BITS, PadState, mask_names
from .metrics import analyze
from .processing import ProcessConfig, describe_process, process
from .session import Clip
from .timing import DeadlineLoop, TimerResolution, freeze_gc, now, set_process_priority

#: Раскладка XInput по индексам кнопок (если SDL не отдал маппинг).
DEFAULT_BUTTONS = (
    "A", "B", "X", "Y", "LB", "RB", "BACK", "START", "GUIDE", "LS", "RS",
    "DU", "DD", "DL", "DR",
)

#: Раскладка XInput по индексам осей (если SDL не отдал маппинг).
DEFAULT_AXES = {"leftx": 0, "lefty": 1, "rightx": 2, "righty": 3, "lt": 4, "rt": 5}


def import_pygame():
    try:
        import pygame  # type: ignore

        return pygame
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Для записи нужен pygame: `pip install pygame` (или pygame-ce). "
            f"Исходная ошибка: {exc}"
        ) from exc


@dataclass
class RecConfig:
    """Настройки записи."""

    duration: float = 0.0  # 0 = до остановки
    rate_hz: float = 250.0
    device: Optional[int] = None
    countdown: float = 3.0
    window: bool = True
    marker_button: str = ""  # кнопка, ставящая метку
    stop_button: str = ""  # кнопка аварийной остановки
    stop_combo: str = ""  # например "BACK+START"
    trigger_deadzone: float = 0.05
    stick_deadzone: float = 0.0  # 0 = писать как есть (чистим потом)
    process: bool = True
    process_cfg: Optional[ProcessConfig] = None
    quiet: bool = False
    live: bool = True
    trim_seconds: float = 0.0  # простая обрезка конца записи


@dataclass
class RecordingResult:
    clip: Clip  # обработанный клип (то, что пойдёт в повтор)
    raw: Clip  # сырая запись
    stats: Dict[str, Any] = field(default_factory=dict)
    process_stats: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    raw_report: Dict[str, Any] = field(default_factory=dict)
    markers: List[Tuple[float, str]] = field(default_factory=list)
    aborted: bool = False
    device_name: str = ""


class PadSource:
    """Источник входных данных: физический геймпад через SDL."""

    def __init__(self, device: Optional[int] = None, *, trigger_deadzone: float = 0.05) -> None:
        self.pygame = import_pygame()
        pg = self.pygame
        pg.init()
        pg.joystick.init()
        self.trigger_deadzone = trigger_deadzone
        self.index = self._pick(device)
        self.joy = pg.joystick.Joystick(self.index)
        self.joy.init()
        self.name = self.joy.get_name()
        self.axes: Dict[str, int] = dict(DEFAULT_AXES)
        self.buttons: Dict[int, int] = {}
        self.trigger_mode: Dict[str, str] = {}  # lt/rt: "signed" | "unsigned" | "missing"
        self._load_mapping()
        self._detect_triggers()

    # -- выбор устройства ---------------------------------------------------
    def _pick(self, device: Optional[int]) -> int:
        pg = self.pygame
        count = pg.joystick.get_count()
        if count == 0:
            raise RuntimeError(
                "Геймпад не найден. Подключите джойстик и запустите заново "
                "(проверить список: autotrack pads)"
            )
        if device is None or device < 0:
            return 0
        if device >= count:
            raise RuntimeError(f"Геймпада с индексом {device} нет (найдено {count})")
        return device

    # -- разбор маппинга ----------------------------------------------------
    def _load_mapping(self) -> None:
        try:
            mapping = self.joy.get_mapping()
        except Exception:
            mapping = ""
        table: Dict[str, str] = {}
        if mapping:
            parts = mapping.split(",")
            for item in parts[2:]:
                if ":" in item:
                    key, value = item.split(":", 1)
                    table[key.strip()] = value.strip()
        axis_keys = ("leftx", "lefty", "rightx", "righty", "lefttrigger", "righttrigger")
        for key in axis_keys:
            target = {"lefttrigger": "lt", "righttrigger": "rt"}.get(key, key)
            value = table.get(key)
            if value and value.startswith("a"):
                try:
                    self.axes[target] = int(value[1:])
                except ValueError:
                    pass
        # кнопки: значение вида b3 или h0.3 (хат) — хаты считаем крестовиной
        for key, value in table.items():
            if not value.startswith("b"):
                continue
            try:
                index = int(value[1:])
            except ValueError:
                continue
            name = {
                "a": "A", "b": "B", "x": "X", "y": "Y",
                "leftshoulder": "LB", "rightshoulder": "RB",
                "back": "BACK", "start": "START", "guide": "GUIDE",
                "leftstick": "LS", "rightstick": "RS",
                "dpup": "DU", "dpdown": "DD", "dpleft": "DL", "dpright": "DR",
                "touchpad": "TOUCHPAD",
            }.get(key)
            if name:
                self.buttons[index] = BUTTON_BITS[name]
        if not self.buttons:  # SDL не дал маппинг — раскладка XInput
            for i, name in enumerate(DEFAULT_BUTTONS):
                self.buttons[i] = BUTTON_BITS[name]

    def _detect_triggers(self) -> None:
        """SDL отдаёт курки как -1..1, но некоторые драйверы — как 0..1."""
        n_axes = self.joy.get_numaxes()
        for key in ("lt", "rt"):
            idx = self.axes.get(key)
            if idx is None or idx >= n_axes:
                self.trigger_mode[key] = "missing"
                continue
            try:
                value = self.joy.get_axis(idx)
            except Exception:
                self.trigger_mode[key] = "missing"
                continue
            self.trigger_mode[key] = "signed" if value < -0.2 else "unsigned"
        # Общая ось Z (типично для DualShock): 4-я ось, LT — в минус, RT — в плюс
        if self.trigger_mode.get("lt") == "missing" and self.trigger_mode.get("rt") == "missing":
            if n_axes == 6 and self.axes["leftx"] == 0:
                self.trigger_mode["lt"] = "combined-"
                self.trigger_mode["rt"] = "combined+"
                self.axes["lt"] = self.axes["rt"] = 4

    # -- чтение -------------------------------------------------------------
    def read(self) -> PadState:
        joy = self.joy
        try:
            lx = joy.get_axis(self.axes["leftx"])
            ly = joy.get_axis(self.axes["lefty"])
            rx = joy.get_axis(self.axes.get("rightx", 2))
            ry = joy.get_axis(self.axes.get("righty", 3))
        except Exception:
            lx = ly = rx = ry = 0.0
        lt = rt = 0.0
        mode_l, mode_r = self.trigger_mode.get("lt", "missing"), self.trigger_mode.get("rt", "missing")
        try:
            if mode_l == "combined-":
                v = joy.get_axis(self.axes["lt"])
                lt = max(0.0, -v)
                rt = max(0.0, v)
            else:
                if mode_l != "missing":
                    v = joy.get_axis(self.axes["lt"])
                    lt = (v + 1.0) / 2.0 if mode_l == "signed" else v
                if mode_r != "missing":
                    v = joy.get_axis(self.axes["rt"])
                    rt = (v + 1.0) / 2.0 if mode_r == "signed" else v
        except Exception:
            lt = rt = 0.0
        if lt < self.trigger_deadzone and mode_l != "missing":
            lt = 0.0
        if rt < self.trigger_deadzone and mode_r != "missing":
            rt = 0.0
        btns = 0
        for index, bit in self.buttons.items():
            try:
                if joy.get_button(index):
                    btns |= 1 << bit
            except Exception:
                continue
        return PadState(
            lx=max(-1.0, min(1.0, lx)),
            ly=max(-1.0, min(1.0, ly)),
            rx=max(-1.0, min(1.0, rx)),
            ry=max(-1.0, min(1.0, ry)),
            lt=max(0.0, min(1.0, lt)),
            rt=max(0.0, min(1.0, rt)),
            btns=btns,
        )

    def close(self) -> None:
        try:
            self.joy.quit()
        except Exception:
            pass
        self.pygame.joystick.quit()


def list_pads() -> List[str]:
    """Список подключённых геймпадов (для команды ``autotrack pads``)."""
    try:
        pg = import_pygame()
    except RuntimeError as exc:
        return [f"pygame недоступен: {exc}"]
    pg.init()
    pg.joystick.init()
    out = []
    for i in range(pg.joystick.get_count()):
        joy = pg.joystick.Joystick(i)
        joy.init()
        axes = joy.get_numaxes()
        out.append(
            f"[{i}] {joy.get_name()} — осей {axes}, кнопок {joy.get_numbuttons()}, "
            f"хатов {joy.get_numhats()}, id {joy.get_guid()[:8]}"
        )
    pg.joystick.quit()
    return out or ["Геймпады не найдены"]


class Recorder:
    """Запись действий геймпада в клип AutoTrack."""

    def __init__(self, config: Optional[RecConfig] = None) -> None:
        self.config = config or RecConfig()

    def record(self, *, keys=None) -> RecordingResult:
        cfg = self.config
        source = PadSource(cfg.device, trigger_deadzone=cfg.trigger_deadzone)
        clip = Clip(
            meta={
                "name": "запись",
                "device": source.name,
                "device_index": source.index,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        window = None
        if cfg.window:
            window = self._open_window(source)
        stop_bit = 1 << BUTTON_BITS[cfg.stop_button.upper()] if cfg.stop_button else 0
        marker_bit = 1 << BUTTON_BITS[cfg.marker_button.upper()] if cfg.marker_button else 0
        combo_bits = 0
        if cfg.stop_combo:
            for name in cfg.stop_combo.replace(",", "+").split("+"):
                name = name.strip().upper()
                if name in BUTTON_BITS:
                    combo_bits |= 1 << BUTTON_BITS[name]
        stats = {}
        loop = DeadlineLoop(cfg.rate_hz, spin_margin=0.00025, precise=True)
        period = 1.0 / cfg.rate_hz
        aborted = False
        paused = False
        markers: List[Tuple[float, str]] = []
        if not cfg.quiet:
            print(f"Геймпад: [{source.index}] {source.name}")
            print(f"Пишем {cfg.rate_hz:.0f} Гц" + (f", {cfg.duration:.1f} с" if cfg.duration else ", до остановки"))
            if marker_bit:
                print(f"  метка — кнопка {cfg.marker_button.upper()}")
            if stop_bit or combo_bits:
                print(f"  стоп  — {cfg.stop_button.upper() or cfg.stop_combo}")
            print("  пауза — пробел/p, стоп — q/Esc, метка — m (в консоли)")
        count = int(cfg.duration * cfg.rate_hz) if cfg.duration else 0
        try:
            with TimerResolution(1):
                set_process_priority("high")
                if not cfg.quiet:
                    self._countdown(cfg.countdown)
                loop.start()
                k = 0
                prev_btns = 0
                with freeze_gc():
                    while True:
                        if keys is not None:
                            ch = keys.poll()
                            if ch:
                                low = ch.lower()
                                if "\x1b" in ch or "q" in low:
                                    aborted = True
                                    break
                                if " " in ch or "p" in low:
                                    paused = not paused
                                    print("  ⏸ пауза" if paused else "  ▶ продолжаем")
                                if "m" in low:
                                    stamp = clip.duration
                                    markers.append((stamp, f"метка {stamp:.2f} с"))
                                    print(f"  ⚑ метка на {stamp:.2f} с")
                        if paused:
                            time.sleep(0.02)
                            loop.t0 = now() - k * period
                            continue
                        state = source.read()
                        # метка кнопкой
                        if marker_bit and (state.btns & marker_bit) and not (prev_btns & marker_bit):
                            stamp = clip.duration
                            markers.append((stamp, f"метка {stamp:.2f} с"))
                            print(f"\n  ⚑ метка на {stamp:.2f} с")
                        prev_btns = state.btns
                        clip.append(k * period, state)
                        if window is not None:
                            self._draw(window, source, state, clip, aborted=False)
                        if stop_bit and (state.btns & stop_bit):
                            aborted = True
                            break
                        if combo_bits and (state.btns & combo_bits) == combo_bits:
                            aborted = True
                            break
                        loop.wait_for(k)
                        k += 1
                        if count and k >= count:
                            break
                        if cfg.live and not cfg.quiet and k % max(1, int(cfg.rate_hz)) == 0:
                            print(
                                f"\r  пишем… {k * period:6.2f} с  L=({state.lx:+.2f},{state.ly:+.2f}) "
                                f"R=({state.rx:+.2f},{state.ry:+.2f}) LT={state.lt:.2f} RT={state.rt:.2f} "
                                f"[{'+'.join(mask_names(state.btns)) or '-':<8}]",
                                end="",
                                flush=True,
                            )
        except KeyboardInterrupt:
            aborted = True
        finally:
            loop_stats = loop.stats.report()
            source.close()
            if window is not None:
                self._close_window(window)
        if not cfg.quiet:
            print("\n" + loop_stats)

        clip.btns = [int(b) for b in clip.btns]
        clip.meta["duration"] = round(clip.duration, 5)
        clip.meta["rate_hz"] = round(clip.rate_hz, 3)
        clip.markers = list(markers) + list(clip.markers)
        raw_report = analyze(clip) if len(clip) > 3 else {}
        raw = clip
        processed = clip
        process_stats: Dict[str, Any] = {}
        if cfg.process and len(clip) > 3:
            res = process(clip, cfg.process_cfg or ProcessConfig(), source_name="запись")
            processed, process_stats = res.clip, res.stats
            if not cfg.quiet:
                print(describe_process(process_stats))
        report = analyze(processed) if len(processed) > 3 else {}
        if cfg.trim_seconds and len(processed) > 3:
            end = max(0.0, processed.duration - cfg.trim_seconds)
            processed = processed.slice(0.0, end)
        return RecordingResult(
            clip=processed,
            raw=raw,
            stats=stats,
            process_stats=process_stats,
            report=report,
            raw_report=raw_report,
            markers=markers,
            aborted=aborted,
            device_name=source.name,
        )

    # -- вспомогательное ----------------------------------------------------
    def _countdown(self, seconds: float) -> None:
        if seconds <= 0:
            return
        print(f"Приготовьтесь: запись начнётся через {seconds:.0f} с")
        end = now() + seconds
        last = None
        while True:
            left = end - now()
            if left <= 0:
                break
            whole = int(math.ceil(left))
            if whole != last:
                print(f"  {whole}…", end="\r", flush=True)
                last = whole
            time.sleep(0.05)
        print("  ● ПИШЕМ" + " " * 20)

    # -- окно ---------------------------------------------------------------
    def _open_window(self, source: PadSource):
        pg = source.pygame
        try:
            screen = pg.display.set_mode((520, 320))
            pg.display.set_caption(f"AutoTrack — запись: {source.name}")
            return screen
        except Exception:
            return None

    def _close_window(self, window) -> None:
        try:
            import pygame

            pygame.display.quit()
        except Exception:
            pass

    def _draw(self, window, source: PadSource, state: PadState, clip: Clip, *, aborted: bool) -> None:
        pg = source.pygame
        try:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    raise KeyboardInterrupt
            window.fill((18, 20, 28))
            font = pg.font.SysFont("consolas", 14)
            cx, cy, r = 120, 175, 70
            for side, (ax, ay, ox) in {"L": ("lx", "ly", -0.0), "R": ("rx", "ry", 0.0)}.items():
                x0 = cx if side == "L" else cx + 260
                pg.draw.circle(window, (60, 66, 84), (x0, cy), r, 2)
                pg.draw.line(window, (60, 66, 84), (x0 - r, cy), (x0 + r, cy), 1)
                pg.draw.line(window, (60, 66, 84), (x0, cy - r), (x0, cy + r), 1)
                px = x0 + int(getattr(state, ax) * r)
                py = cy + int(getattr(state, ay) * r)
                pg.draw.circle(window, (90, 200, 120), (px, py), 8)
            txt = font.render(f"{state.describe()}", True, (220, 224, 235))
            window.blit(txt, (10, 8))
            info = font.render(f"записано {clip.duration:7.2f} с  ({len(clip)} сэмплов)", True, (200, 200, 210))
            window.blit(info, (10, 290))
            bar_w = 500
            frac = 0.0
            if self.config.duration:
                frac = min(1.0, clip.duration / self.config.duration)
            pg.draw.rect(window, (40, 44, 58), (10, 270, bar_w, 8))
            pg.draw.rect(window, (90, 200, 120), (10, 270, int(bar_w * frac), 8))
            pg.display.flip()
        except Exception:
            pass


__all__ = ["RecConfig", "RecordingResult", "Recorder", "PadSource", "list_pads"]

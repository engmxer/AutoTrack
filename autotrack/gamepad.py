"""Виртуальный геймпад (вывод) + утилиты состояния.

Основной бэкенд — ``vgamepad`` (драйвер ViGEmBus): система видит виртуальный
Xbox 360 / XInput-контроллер, который и «нажимает» за игрока.

Есть три реализации:

* :class:`XInputPad`  — боевой вывод через vgamepad (виртуальный геймпад);
* :class:`DryRunPad`  — ничего не отправляет, но пишет телеметрию (тест/отладка);
* :class:`NullPad`    — заглушка.

Ключевые моменты производительности:

* «быстрый путь» — прямая запись полей ``report`` + один ``update()``;
* кнопки отправляются только при изменении маски;
* курки квантуются в 0..255 и тоже отправляются только при изменении значения;
* при выходе (в т.ч. по Ctrl+C) всё гарантированно отпускается.
"""

from __future__ import annotations

import atexit
import time
from typing import Dict, Optional

from .frame import PadState, clamp

#: Соответствие наших имён кнопок битам XInput (wButtons).
XINPUT_BITS: Dict[str, int] = {
    "DU": 0x0001,
    "DD": 0x0002,
    "DL": 0x0004,
    "DR": 0x0008,
    "START": 0x0010,
    "BACK": 0x0020,
    "LS": 0x0040,
    "RS": 0x0080,
    "LB": 0x0100,
    "RB": 0x0200,
    "GUIDE": 0x0400,
    "A": 0x1000,
    "B": 0x2000,
    "X": 0x4000,
    "Y": 0x8000,
}

#: Соответствие наших имён константам vgamepad (резервный, «медленный» путь).
VGPAD_BUTTON_NAMES: Dict[str, str] = {
    "A": "XUSB_GAMEPAD_A",
    "B": "XUSB_GAMEPAD_B",
    "X": "XUSB_GAMEPAD_X",
    "Y": "XUSB_GAMEPAD_Y",
    "LB": "XUSB_GAMEPAD_LEFT_SHOULDER",
    "RB": "XUSB_GAMEPAD_RIGHT_SHOULDER",
    "LS": "XUSB_GAMEPAD_LEFT_THUMB",
    "RS": "XUSB_GAMEPAD_RIGHT_THUMB",
    "BACK": "XUSB_GAMEPAD_BACK",
    "START": "XUSB_GAMEPAD_START",
    "GUIDE": "XUSB_GAMEPAD_GUIDE",
    "DU": "XUSB_GAMEPAD_DPAD_UP",
    "DD": "XUSB_GAMEPAD_DPAD_DOWN",
    "DL": "XUSB_GAMEPAD_DPAD_LEFT",
    "DR": "XUSB_GAMEPAD_DPAD_RIGHT",
}


def xinput_mask(btns: int) -> int:
    """Наша маска кнопок -> маска XInput ``wButtons``."""
    from .frame import BUTTON_BITS

    out = 0
    for name, bit in BUTTON_BITS.items():
        if btns & (1 << bit):
            xi = XINPUT_BITS.get(name)
            if xi:
                out |= xi
    return out


class VirtualPad:
    """Интерфейс вывода."""

    name = "virtual"
    available = True

    def write(self, state: PadState) -> None:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def reset(self) -> None:
        try:
            self.write(PadState())
        except Exception:
            pass

    def close(self) -> None:
        self.reset()

    def __enter__(self) -> "VirtualPad":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullPad(VirtualPad):
    """Заглушка: ничего не делает (для тестов и «сухого» прогона отчётов)."""

    name = "null"

    def write(self, state: PadState) -> None:
        self.last = state


class DryRunPad(VirtualPad):
    """Пишет всё во внутренний лог: удобно проверять тайминги без игры.

    Логирует только изменения, чтобы файл не разрастался.
    """

    name = "dry-run"

    def __init__(self, sample_every: int = 0) -> None:
        self.rows: list[tuple[float, PadState]] = []
        self.sample_every = sample_every
        self._k = 0
        self._last: Optional[PadState] = None

    def write(self, state: PadState) -> None:
        self._k += 1
        changed = state != self._last
        sampled = self.sample_every and self._k % self.sample_every == 0
        if changed or sampled:
            self.rows.append((time.perf_counter(), state))
            self._last = state


class XInputPad(VirtualPad):
    """Виртуальный Xbox 360 геймпад через ``vgamepad`` + ViGEmBus.

    Два режима записи состояния:

    * «быстрый» — прямая запись полей отчёта (``wButtons``, ``bLeftTrigger``,
      ``sThumbLX``…) и один ``update()`` на кадр. Экономит вызовы API, что
      важно при 250 кадрах/с;
    * «медленный» (резерв) — публичные методы ``left_joystick`` и т.д.

    Если состояние не изменилось (нейтраль на паузе), отправка пропускается.
    """

    name = "xinput(x360)"

    #: имена полей отчёта XInput в vgamepad
    _FIELD_CANDIDATES = {
        "lx": ("sThumbLX", "s_lx"),
        "ly": ("sThumbLY", "s_ly"),
        "rx": ("sThumbRX", "s_rx"),
        "ry": ("sThumbRY", "s_ry"),
        "lt": ("bLeftTrigger", "b_left_trigger"),
        "rt": ("bRightTrigger", "b_right_trigger"),
        "btns": ("wButtons",),
    }

    def __init__(self, *, deadzone: float = 0.0, invert_y: bool = True) -> None:
        try:
            import vgamepad as vg
        except Exception as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "Не удалось импортировать vgamepad. Нужны: `pip install vgamepad` и драйвер "
                "ViGEmBus (https://github.com/nefarius/ViGEmBus/releases). "
                f"Исходная ошибка: {exc}"
            ) from exc
        self._vg = vg
        self._deadzone = max(0.0, float(deadzone))
        self._invert_y = bool(invert_y)
        self.pad = vg.VX360Gamepad()
        self._fields: Dict[str, str] = {}
        self._detect_fields()
        self._fast = bool(self._fields)
        self._resolved = {}
        from .frame import BUTTON_BITS

        for name, attr in VGPAD_BUTTON_NAMES.items():
            const = getattr(vg.XUSB_BUTTON, attr, None)
            if const is not None:
                self._resolved[BUTTON_BITS[name]] = const
        self._buttons = -1  # «ничего не отправлено» -> первое состояние уйдёт
        self._triggers = (-1, -1)
        self._sticks = (None, None, None, None)
        atexit.register(self.close)

    # -- внутреннее ---------------------------------------------------------
    def _detect_fields(self) -> None:
        """Проверить, какие поля есть в отчёте текущей версии vgamepad."""
        try:
            report = self.pad.report
        except Exception:
            return
        for key, names in self._FIELD_CANDIDATES.items():
            for name in names:
                if hasattr(report, name):
                    self._fields[key] = name
                    break

    @staticmethod
    def _axis_to_int(value: float) -> int:
        return int(clamp(value, -1.0, 1.0) * 32767.0)

    def _write_fast(self, state: PadState) -> None:
        report = self.pad.report
        dead = self._deadzone
        sgn = -1.0 if self._invert_y else 1.0
        lx, ly = state.lx, sgn * state.ly
        rx, ry = state.rx, sgn * state.ry
        if dead:
            lx = 0.0 if abs(lx) < dead else lx
            ly = 0.0 if abs(ly) < dead else ly
            rx = 0.0 if abs(rx) < dead else rx
            ry = 0.0 if abs(ry) < dead else ry
        sticks = (
            self._axis_to_int(lx), self._axis_to_int(ly),
            self._axis_to_int(rx), self._axis_to_int(ry),
        )
        lt = int(clamp(state.lt, 0.0, 1.0) * 255.0)
        rt = int(clamp(state.rt, 0.0, 1.0) * 255.0)
        buttons = xinput_mask(state.btns)

        if sticks == self._sticks and (lt, rt) == self._triggers and buttons == self._buttons:
            return  # нечего отправлять
        f = self._fields
        if sticks != self._sticks:
            setattr(report, f["lx"], sticks[0])
            setattr(report, f["ly"], sticks[1])
            setattr(report, f["rx"], sticks[2])
            setattr(report, f["ry"], sticks[3])
            self._sticks = sticks
        if (lt, rt) != self._triggers:
            setattr(report, f["lt"], lt)
            setattr(report, f["rt"], rt)
            self._triggers = (lt, rt)
        if buttons != self._buttons:
            setattr(report, f["btns"], buttons)
            self._buttons = buttons
        self.pad.update()

    def _write_api(self, state: PadState) -> None:
        """Резервный путь через публичный API vgamepad."""
        pad = self.pad
        sgn = -1.0 if self._invert_y else 1.0
        pad.left_joystick(self._axis_to_int(state.lx), self._axis_to_int(sgn * state.ly))
        pad.right_joystick(self._axis_to_int(state.rx), self._axis_to_int(sgn * state.ry))
        pad.left_trigger(int(clamp(state.lt, 0.0, 1.0) * 255.0))
        pad.right_trigger(int(clamp(state.rt, 0.0, 1.0) * 255.0))
        want = xinput_mask(state.btns)
        changed = want ^ (0 if self._buttons < 0 else self._buttons)
        for bit, const in self._resolved.items():
            xi = XINPUT_BITS.get(_bit_name(bit))
            if not xi or not (changed & xi):
                continue
            if want & xi:
                pad.press_button(button=const)
            else:
                pad.release_button(button=const)
        self._buttons = want
        pad.update()

    def write(self, state: PadState) -> None:
        if self._fast:
            try:
                self._write_fast(state)
                return
            except Exception:
                # поля отчёта могут отличаться в других версиях vgamepad
                self._fast = False
        self._write_api(state)

    def reset(self) -> None:
        self._buttons = -1
        self._triggers = (-1, -1)
        self._sticks = (None, None, None, None)
        try:
            self.pad.reset()
        except Exception:
            pass
        # отправляем нейтраль заново
        try:
            self.write(PadState())
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.reset()
        except Exception:
            pass


def _bit_name(bit: int) -> str:
    from .frame import BUTTON_NAMES

    return BUTTON_NAMES.get(bit, "")


def make_pad(kind: str = "xinput", *, deadzone: float = 0.0, invert_y: bool = True) -> VirtualPad:
    """Фабрика выходного устройства: ``xinput`` | ``dry`` | ``null``."""
    kind = (kind or "xinput").lower()
    if kind in ("xinput", "x360", "vgamepad", "gamepad"):
        return XInputPad(deadzone=deadzone, invert_y=invert_y)
    if kind in ("dry", "dry-run", "test", "none"):
        return DryRunPad()
    if kind == "null":
        return NullPad()
    raise ValueError(f"неизвестный бэкенд вывода: {kind!r}")


__all__ = [
    "VirtualPad",
    "XInputPad",
    "DryRunPad",
    "NullPad",
    "make_pad",
    "xinput_mask",
    "XINPUT_BITS",
]

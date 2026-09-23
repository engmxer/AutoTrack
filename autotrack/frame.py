"""Каноническое представление состояния геймпада.

Соглашения по осям (совпадают с SDL / физическим джойстиком):

* ``lx, ly``  — левый стик,  диапазон -1.0 .. 1.0, **+Y = вниз**;
* ``rx, ry``  — правый стик, диапазон -1.0 .. 1.0, **+Y = вниз**;
* ``lt, rt``  — курки,      диапазон  0.0 .. 1.0;
* ``btns``    — битовая маска кнопок (см. :data:`BUTTONS`).

Инверсия Y делается только на границе вывода (виртуальный геймпад XInput),
поэтому записанные данные всегда лежат в одной и той же системе координат.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, NamedTuple

# ---------------------------------------------------------------------------
# Кнопки
# ---------------------------------------------------------------------------

#: Имя кнопки -> номер бита в маске.
BUTTON_BITS: Dict[str, int] = {
    "A": 0,
    "B": 1,
    "X": 2,
    "Y": 3,
    "LB": 4,
    "RB": 5,
    "LS": 6,
    "RS": 7,
    "BACK": 8,
    "START": 9,
    "GUIDE": 10,
    "DU": 11,
    "DD": 12,
    "DL": 13,
    "DR": 14,
    "TOUCHPAD": 15,
    "CAPTURE": 16,
    "P1": 17,
    "P2": 18,
    "P3": 19,
    "P4": 20,
    "MISC1": 21,
}

BUTTON_NAMES: Dict[int, str] = {bit: name for name, bit in BUTTON_BITS.items()}

#: Имя кнопки из SDL GameController -> наше имя.
SDL_BUTTON_MAP: Dict[str, str] = {
    "a": "A",
    "b": "B",
    "x": "X",
    "y": "Y",
    "leftshoulder": "LB",
    "rightshoulder": "RB",
    "leftstick": "LS",
    "rightstick": "RS",
    "back": "BACK",
    "start": "START",
    "guide": "GUIDE",
    "dpup": "DU",
    "dpdown": "DD",
    "dpleft": "DL",
    "dpright": "DR",
    "touchpad": "TOUCHPAD",
    "capture": "CAPTURE",
    "misc1": "MISC1",
    "misc2": "MISC1",
    "paddle1": "P1",
    "paddle2": "P2",
    "paddle3": "P3",
    "paddle4": "P4",
}

#: Кнопки, которые всегда имеет смысл показывать в отчётах/статистике.
COMMON_BUTTONS = (
    "A", "B", "X", "Y", "LB", "RB", "LS", "RS", "BACK", "START",
    "DU", "DD", "DL", "DR",
)

AXES = ("lx", "ly", "rx", "ry", "lt", "rt")
STICK_AXES = {"L": ("lx", "ly"), "R": ("rx", "ry")}
TRIGGER_AXES = ("lt", "rt")


def bit(name: str) -> int:
    """Бит по имени кнопки (регистр не важен)."""
    try:
        return 1 << BUTTON_BITS[name.upper()]
    except KeyError:
        raise KeyError(f"неизвестная кнопка: {name!r}") from None


def mask(names: Iterable[str]) -> int:
    """Маска по последовательности имён кнопок."""
    out = 0
    for name in names:
        out |= bit(name)
    return out


def mask_names(value: int) -> list[str]:
    """Имена нажатых кнопок по маске (в стабильном порядке)."""
    out = []
    remaining = value
    for name, idx in BUTTON_BITS.items():
        b = 1 << idx
        if remaining & b:
            out.append(name)
            remaining &= ~b
    return out


def clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


class PadState(NamedTuple):
    """Одно состояние геймпада."""

    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    btns: int = 0

    # -- удобные производные ------------------------------------------------
    @property
    def left_mag(self) -> float:
        return math.hypot(self.lx, self.ly)

    @property
    def right_mag(self) -> float:
        return math.hypot(self.rx, self.ry)

    def axis(self, name: str) -> float:
        return getattr(self, name)

    def with_buttons(self, btns: int) -> "PadState":
        return self._replace(btns=btns)

    def neutral(self) -> bool:
        return (
            self.lx == 0.0
            and self.ly == 0.0
            and self.rx == 0.0
            and self.ry == 0.0
            and self.lt == 0.0
            and self.rt == 0.0
            and self.btns == 0
        )

    def as_dict(self) -> dict:
        return {
            "lx": self.lx,
            "ly": self.ly,
            "rx": self.rx,
            "ry": self.ry,
            "lt": self.lt,
            "rt": self.rt,
            "btns": self.btns,
            "buttons": mask_names(self.btns),
        }

    def describe(self) -> str:
        names = mask_names(self.btns)
        return (
            f"L=({self.lx:+.3f},{self.ly:+.3f}) R=({self.rx:+.3f},{self.ry:+.3f}) "
            f"LT={self.lt:.2f} RT={self.rt:.2f} "
            f"[{'+'.join(names) if names else '-'}]"
        )


NEUTRAL = PadState()

__all__ = [
    "PadState",
    "NEUTRAL",
    "AXES",
    "STICK_AXES",
    "TRIGGER_AXES",
    "BUTTON_BITS",
    "BUTTON_NAMES",
    "COMMON_BUTTONS",
    "SDL_BUTTON_MAP",
    "bit",
    "mask",
    "mask_names",
    "clamp",
]

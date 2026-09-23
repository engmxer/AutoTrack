"""Фильтры для «вычисления плавности» стиков.

Содержит:

* :class:`OneEuroFilter` — адаптивный фильтр (Casiez et al., 2012). Именно он
  используется для сглаживания стиков: на медленных движениях даёт сильное
  сглаживание (убирает дрожание), на быстрых — почти не тормозит.
* :class:`AngleFilter` — сглаживание **направления** стика отдельно от
  амплитуды: угол разворачивается в непрерывную последовательность,
  фильтруется One Euro и собирается обратно в (x, y). Это и есть «вычисление
  плавности направления грибков»: дрожание угла давится, а радиус поворота
  сохраняется.
* :func:`despike` — медиана по 3 точкам (удаление одиночных выбросов
  дешёвых стиков).
* :func:`unwrap_angle`, :func:`moving_average`, :class:`StickSmoother`.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple



def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Одномерный One Euro filter.

    ``min_cutoff`` — частота среза в покое (меньше = плавнее),
    ``beta`` — насколько фильтр «отпускает» сигнал при быстром движении.
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x", "_dx", "initialized")

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.03, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: float = 0.0
        self._dx: float = 0.0
        self.initialized = False

    def reset(self, x: float = 0.0) -> None:
        self._x = float(x)
        self._dx = 0.0
        self.initialized = True

    def __call__(self, x: float, dt: float) -> float:
        x = float(x)
        if dt <= 0:
            dt = 1e-3
        if not self.initialized:
            self.reset(x)
            return self._x
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x
        self._x, self._dx = x_hat, dx_hat
        return x_hat

    @classmethod
    def apply_series(
        cls, values: Sequence[float], dt: float, *, min_cutoff: float, beta: float
    ) -> List[float]:
        f = cls(min_cutoff, beta)
        return [f(v, dt) for v in values]


class EMA:
    """Простое экспоненциальное сглаживание (для отчётов и грубых проверок)."""

    __slots__ = ("alpha", "_y", "initialized")

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = float(alpha)
        self._y = 0.0
        self.initialized = False

    def __call__(self, x: float) -> float:
        if not self.initialized:
            self._y = float(x)
            self.initialized = True
        else:
            self._y += self.alpha * (float(x) - self._y)
        return self._y


def unwrap_angle(values: Iterable[float]) -> List[float]:
    """Развернуть угол (радианы) в непрерывную последовательность без скачков ±2π."""
    out: List[float] = []
    offset = 0.0
    prev: Optional[float] = None
    for a in values:
        if prev is not None:
            d = a - prev
            if d > math.pi:
                offset -= 2.0 * math.pi
            elif d < -math.pi:
                offset += 2.0 * math.pi
        out.append(a + offset)
        prev = a
    return out


def wrap_angle(a: float) -> float:
    """Привести угол к диапазону (-π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_series(xs: Sequence[float], ys: Sequence[float]) -> List[float]:
    """Мгновенное направление стика (радианы) для каждой точки."""
    return [math.atan2(y, x) for x, y in zip(xs, ys)]


def despike(values: Sequence[float], window: int = 3) -> List[float]:
    """Медианный фильтр (по умолчанию окно 3) — убирает одиночные выбросы."""
    if window <= 1 or len(values) < 3:
        return list(values)
    half = window // 2
    n = len(values)
    out: List[float] = []
    src = list(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = sorted(src[lo:hi])
        out.append(chunk[len(chunk) // 2])
    return out


def zero_phase_stick(
    lx: Sequence[float],
    ly: Sequence[float],
    spacing_hz: float,
    cutoff_hz: float = 10.0,
    *,
    order: int = 2,
    deadzone: float = 0.02,
) -> Tuple[List[float], List[float]]:
    """Сгладить готовый маршрут **без задержки** (фильтр «вперёд-назад»).

    Запись известна целиком до старта, поэтому можно применить нуль-фазовый
    фильтр: дрожание руки убирается, а вывода по времени не сдвигается —
    повтор не отстаёт от записи (в отличие от сглаживания «на лету», которое
    даёт постоянное отставание ``~1/(2π·f_c)`` секунд).

    Амплитуда и **развёрнутый угол** фильтруются отдельно: ровные дуги не
    «скручиваются» у центра, где направление физически не определено.
    """
    from .dsp import lowpass

    n = min(len(lx), len(ly))
    if n < 8 or cutoff_hz <= 0.0 or cutoff_hz >= spacing_hz * 0.5:
        return list(lx[:n]), list(ly[:n])
    mag = [math.hypot(lx[i], ly[i]) for i in range(n)]
    # угол: пока амплитуда в «нуле», направление не определено — держим последнее
    ang: List[float] = []
    last = 0.0
    prev: Optional[float] = None
    for i in range(n):
        if mag[i] >= deadzone:
            a = math.atan2(ly[i], lx[i])
            if prev is not None:
                while a - prev > math.pi:
                    a -= 2.0 * math.pi
                while a - prev < -math.pi:
                    a += 2.0 * math.pi
            prev = a
            last = a
        ang.append(last)
    first_valid = next((i for i in range(n) if mag[i] >= deadzone), None)
    if first_valid is not None:
        head = ang[first_valid]
        for i in range(first_valid):
            ang[i] = head
    fm = lowpass(mag, cutoff_hz, spacing_hz, order)
    fa = lowpass(ang, cutoff_hz, spacing_hz, order)
    out_x: List[float] = []
    out_y: List[float] = []
    for i in range(n):
        m = fm[i] if fm[i] > 0.0 else 0.0
        a = fa[i]
        out_x.append(m * math.cos(a))
        out_y.append(m * math.sin(a))
    return out_x, out_y


class StickSmoother:
    """Сглаживание одного стика: амплитуда + направление отдельно.

    Режимы:

    * ``xy`` — фильтруем x и y напрямую (безопасно, но чуть «заваливает» углы
      при быстром вращении);
    * ``polar`` — фильтруем амплитуду и **развёрнутый угол**, затем собираем
      обратно. Даёт заметно более ровные дуги: дрожание направления
      подавляется независимо от радиуса.

    ``radius_deadzone`` не даёт углу «дёргаться» у центра: пока амплитуда
    меньше порога, направление держится последним валидным.
    """

    __slots__ = (
        "mode",
        "deadzone",
        "_fx",
        "_fy",
        "_fm",
        "_fa",
        "_last_angle",
        "_has_angle",
        "applied",
    )

    def __init__(
        self,
        *,
        spacing_hz: float,
        min_cutoff: float = 1.6,
        beta: float = 0.02,
        beta_angle: float = 0.012,
        mode: str = "polar",
        deadzone: float = 0.05,
    ) -> None:
        self.mode = mode
        self.deadzone = float(deadzone)
        self._fx = OneEuroFilter(min_cutoff, beta)
        self._fy = OneEuroFilter(min_cutoff, beta)
        self._fm = OneEuroFilter(min_cutoff, beta)
        # для угла производную оцениваем медленнее (d_cutoff ниже): иначе
        # адаптивный срез сам начинает «дрожать» и добавляет угловое ускорение
        self._fa = OneEuroFilter(min_cutoff, beta_angle, d_cutoff=0.4)
        self._last_angle = 0.0
        self._has_angle = False
        self.applied = 0

    def reset(self, x: float = 0.0, y: float = 0.0) -> None:
        self._fx.reset(x)
        self._fy.reset(y)
        self._fm.reset(math.hypot(x, y))
        if math.hypot(x, y) > self.deadzone:
            self._last_angle = math.atan2(y, x)
            self._has_angle = True
        else:
            self._has_angle = False

    def __call__(self, x: float, y: float, dt: float) -> Tuple[float, float]:
        if self.mode == "off":
            return x, y
        if not self._fm.initialized:
            self.reset(x, y)
            self.applied += 1
            return x, y
        if self.mode == "xy":
            self.applied += 1
            return self._fx(x, dt), self._fy(y, dt)
        mag = math.hypot(x, y)
        mag_f = self._fm(mag, dt)
        if mag > self.deadzone:
            angle = math.atan2(y, x)
            if self._has_angle:
                # разворачиваем к предыдущему углу, чтобы не было скачка через ±π
                while angle - self._last_angle > math.pi:
                    angle -= 2.0 * math.pi
                while angle - self._last_angle < -math.pi:
                    angle += 2.0 * math.pi
            angle_f = self._fa(angle, dt) if self._has_angle else angle
            self._last_angle = angle_f
            self._has_angle = True
        else:
            angle_f = self._last_angle
            mag_f = min(mag_f, self.deadzone)
        self.applied += 1
        return mag_f * math.cos(angle_f), mag_f * math.sin(angle_f)

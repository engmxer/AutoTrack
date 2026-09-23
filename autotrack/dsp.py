"""Цифровая обработка сигнала для AutoTrack (без обязательных зависимостей).

Всё, что нужно для честного анализа и мягкой пост-обработки:

* :func:`lowpass`, :func:`highpass`, :func:`bandpass` — Баттерворт, **нуль-фазовый**
  (вперёд-назад, ``filtfilt``): сигнал не сдвигается по времени;
* :func:`moving_average` — быстрое сглаживание;
* :func:`median_filter` — удаление одиночных выбросов;
* :func:`resample` — пересчёт на другую сетку времени (линейно/кубически);
* :func:`detect_outliers` — поиск выбросов по остатку от фильтра.

Если установлен ``scipy`` — используется он (быстро и точно). Если нет —
работает собственная реализация на биквадах (RBJ-коэффициенты), чисто на
Python + опционально numpy.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

try:  # необязательное ускорение
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:  # необязательный scipy
    from scipy import signal as _sp_signal
except Exception:  # pragma: no cover
    _sp_signal = None


# ---------------------------------------------------------------------------
# Простые фильтры
# ---------------------------------------------------------------------------


def moving_average(values: Sequence[float], win: int) -> List[float]:
    """Скользящее среднее вперёд и назад (нуль-фазовое)."""
    n = len(values)
    if win <= 1 or n < 3:
        return list(values)
    win = min(win, n)
    half = win // 2
    if _np is not None:
        x = _np.asarray(values, dtype=float)
        pad = _np.pad(x, half, mode="edge")
        c = _np.cumsum(_np.concatenate(([0.0], pad)))
        avg = (c[win:] - c[:-win]) / win
        pad2 = _np.pad(avg, half, mode="edge")
        c2 = _np.cumsum(_np.concatenate(([0.0], pad2)))
        avg2 = (c2[win:] - c2[:-win]) / win
        out = list(avg2[:n])
        while len(out) < n:
            out.append(out[-1] if out else 0.0)
        return out
    fwd: List[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        fwd.append(sum(values[lo:hi]) / (hi - lo))
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(fwd[lo:hi]) / (hi - lo))
    return out


def median_filter(values: Sequence[float], win: int = 3) -> List[float]:
    """Медианный фильтр — убирает одиночные выбросы, сохраняя «ступеньки»."""
    n = len(values)
    if win <= 1 or n < 3:
        return list(values)
    half = win // 2
    src = list(values)
    out: List[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = sorted(src[lo:hi])
        out.append(chunk[len(chunk) // 2])
    return out


def window_for_cutoff(fs: float, cutoff_hz: float, lo: int = 3, hi: int = 151) -> int:
    """Окно скользящего среднего, дающее срез примерно на ``cutoff_hz``."""
    if fs <= 0 or cutoff_hz <= 0:
        return lo
    win = int(round(0.443 * fs / cutoff_hz))
    return max(lo, min(hi, win)) | 1


# ---------------------------------------------------------------------------
# Баттерворт: коэффициенты и применение
# ---------------------------------------------------------------------------


def _biquads(kind: str, cutoff: float, fs: float, order: int = 2) -> List[Tuple[float, ...]]:
    """Каскад биквадов Баттерворта (RBJ), ``kind`` = low/high/band."""
    if fs <= 0:
        raise ValueError("fs должен быть > 0")
    nyq = fs / 2.0
    n_sections = max(1, order // 2)
    out: List[Tuple[float, ...]] = []
    for k in range(n_sections):
        if order == 2:
            q = 1.0 / math.sqrt(2.0)
        else:
            # Баттерворт чётного порядка: Q пар полюсов
            theta = math.pi / (2.0 * order) * (2.0 * k + 1)
            q = 1.0 / (2.0 * math.sin(theta))
        if kind == "band":
            f0, bw = cutoff  # type: ignore[misc]
            w0 = 2.0 * math.pi * f0 / fs
            alpha = math.sin(w0) * math.sinh(math.log(2.0) / 2.0 * bw * w0 / math.sin(w0))
            b = (alpha, 0.0, -alpha)
            a = (1.0 + alpha, -2.0 * math.cos(w0), 1.0 - alpha)
        else:
            w0 = 2.0 * math.pi * min(cutoff, nyq * 0.999) / fs
            sn, cs = math.sin(w0), math.cos(w0)
            alpha = sn / (2.0 * q)
            if kind == "low":
                b = ((1.0 - cs) / 2.0, 1.0 - cs, (1.0 - cs) / 2.0)
            else:
                b = ((1.0 + cs) / 2.0, -(1.0 + cs), (1.0 + cs) / 2.0)
            a = (1.0 + alpha, -2.0 * cs, 1.0 - alpha)
        b = tuple(v / a[0] for v in b)
        a = tuple(v / a[0] for v in a)
        out.append((b[0], b[1], b[2], 1.0, a[1], a[2]))
    return out


def _dc_gain(sos: Tuple[float, ...]) -> float:
    """Коэффициент передачи на постоянном токе (для начальных условий)."""
    b0, b1, b2, _a0, a1, a2 = sos
    denom = 1.0 + a1 + a2
    return (b0 + b1 + b2) / denom if abs(denom) > 1e-12 else 0.0


def _apply_biquad(x, sos: Tuple[float, ...], init: float = 0.0) -> List[float]:
    """Один биквад. ``init`` — установившееся значение входа на старте.

    Благодаря инициализации состояния «по постоянному току» в начале и конце
    нет холодного переходного процесса — именно из-за него края записи раньше
    выходили «шумными» после фильтрации.
    """
    b0, b1, b2, _a0, a1, a2 = sos
    x1 = x2 = init
    y1 = y2 = init * _dc_gain(sos)
    out: List[float] = []
    append = out.append
    for v in x:
        y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, v
        y2, y1 = y1, y
        append(y)
    return out


def _apply_cascade(x: List[float], sos_list: List[Tuple[float, ...]], init: float) -> List[float]:
    ss = init
    for sos in sos_list:
        x = _apply_biquad(x, sos, init=ss)
        ss *= _dc_gain(sos)
    return x


def _filtfilt_manual(x: Sequence[float], sos_list: List[Tuple[float, ...]]) -> List[float]:
    """Нуль-фазовый фильтр: прямой проход, реверс, обратный проход.

    Края не «достраиваются» отражением (у высокочастотного сигнала отражение
    рождает низкочастотный артефакт до 2.0 по амплитуде!) — вместо этого
    фильтр запускается сразу в установившемся режиме (по постоянному току),
    поэтому переходного процесса на краях нет.
    """
    n = len(x)
    if n == 0:
        return []
    y = _apply_cascade(list(x), sos_list, init=x[0])
    y.reverse()
    y = _apply_cascade(y, sos_list, init=y[0])
    y.reverse()
    return y


def filter_signal(x: Sequence[float], kind: str, cutoff, fs: float, order: int = 4) -> List[float]:
    """Нуль-фазовая фильтрация (Баттерворт). ``kind``: low/high/band."""
    n = len(x)
    if n < 8 or fs <= 0:
        return list(x)
    if _sp_signal is not None:
        try:
            if kind == "band":
                lo, hi = cutoff  # type: ignore[misc]
                sos = _sp_signal.butter(max(2, order), [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
            else:
                btype = "lowpass" if kind == "low" else "highpass"
                sos = _sp_signal.butter(max(2, order), cutoff / (fs / 2), btype=btype, output="sos")
            y = _sp_signal.sosfiltfilt(sos, _np.asarray(x, dtype=float)) if _np is not None else None
            if y is not None:
                return [float(v) for v in y]
        except Exception:
            pass
    nyq = fs / 2.0
    if kind != "band" and cutoff >= nyq * 0.99:
        return list(x)
    sos_list = _biquads(kind, cutoff, fs, order)
    return _filtfilt_manual(x, sos_list)


def lowpass(x: Sequence[float], cutoff_hz: float, fs: float, order: int = 4) -> List[float]:
    """Нуль-фазовый ФНЧ Баттерворта."""
    return filter_signal(x, "low", cutoff_hz, fs, order)


def highpass(x: Sequence[float], cutoff_hz: float, fs: float, order: int = 4) -> List[float]:
    """Нуль-фазовый ФВЧ Баттерворта (для оценки дрожания)."""
    return filter_signal(x, "high", cutoff_hz, fs, order)


def bandpass(x: Sequence[float], lo_hz: float, hi_hz: float, fs: float, order: int = 4) -> List[float]:
    """Нуль-фазовый полосовой фильтр."""
    return filter_signal(x, "band", (lo_hz, hi_hz), fs, order)


# ---------------------------------------------------------------------------
# Пересэмплирование и поиск выбросов
# ---------------------------------------------------------------------------


def _catmull_rom(a: float, b: float, c: float, d: float, f: float) -> float:
    f2 = f * f
    f3 = f2 * f
    return 0.5 * (
        (2.0 * b) + (-a + c) * f + (2.0 * a - 5.0 * b + 4.0 * c - d) * f2 + (-a + 3.0 * b - 3.0 * c + d) * f3
    )


def resample(
    values: Sequence[float],
    t_src: Sequence[float],
    t_dst: Sequence[float],
    *,
    interp: str = "linear",
) -> List[float]:
    """Пересэмплировать ряд на новую сетку времени."""
    n = len(values)
    if n == 0:
        return [0.0] * len(t_dst)
    if n == 1:
        return [values[0]] * len(t_dst)
    out: List[float] = []
    i = 0
    for tt in t_dst:
        while i + 1 < n and t_src[i + 1] <= tt:
            i += 1
        j = min(i, n - 1)
        if j + 1 >= n or t_src[j + 1] <= t_src[j]:
            out.append(values[j])
            continue
        span = t_src[j + 1] - t_src[j]
        f = 0.0 if span <= 0 else (tt - t_src[j]) / span
        f = 0.0 if f < 0 else (1.0 if f > 1 else f)
        if interp == "cubic":
            out.append(
                _catmull_rom(
                    values[max(0, j - 1)], values[j], values[j + 1], values[min(n - 1, j + 2)], f
                )
            )
        elif interp == "hold" or f == 0.0:
            out.append(values[j])
        else:
            out.append(values[j] + (values[j + 1] - values[j]) * f)
    return out


def uniform_grid(duration: float, rate_hz: float) -> List[float]:
    """Ровная сетка времени 0..duration с шагом 1/rate."""
    if rate_hz <= 0:
        raise ValueError("rate_hz должен быть > 0")
    step = 1.0 / rate_hz
    count = max(1, int(round(duration * rate_hz))) + 1
    return [i * step for i in range(count)]


def detect_outliers(values: Sequence[float], cutoff_hz: float, fs: float, k: float = 4.0) -> List[int]:
    """Индексы выбросов: где остаток от фильтра больше ``k`` СКО остатка."""
    if len(values) < 16:
        return []
    smooth = lowpass(values, cutoff_hz, fs)
    res = [a - b for a, b in zip(values, smooth)]
    sd = math.sqrt(sum(v * v for v in res) / len(res))
    if sd <= 0:
        return []
    return [i for i, v in enumerate(res) if abs(v) > k * sd]


__all__ = [
    "moving_average",
    "median_filter",
    "window_for_cutoff",
    "lowpass",
    "highpass",
    "bandpass",
    "filter_signal",
    "resample",
    "uniform_grid",
    "detect_outliers",
]

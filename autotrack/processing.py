"""Пост-обработка записи: чистка, сглаживание, ровная сетка времени.

Порядок операций (после записи и/или перед воспроизведением):

1. **выбросы** — медианный фильтр по 3 точкам снимает одиночные «иглы»;
2. **сглаживание** — нуль-фазовый ФНЧ Баттерворта (по умолчанию 14 Гц):
   дрожание руки уходит, траектория машины (доли герца) остаётся;
3. **обрезка простоя** — «нейтральное» начало/конец, если попросили;
4. **ровная сетка** — пересчёт на фиксированную частоту (по умолчанию 120 Гц):
   так воспроизведение не зависит от разброса таймингов записи;
5. **согласование импульсов** — кнопки/курки, которые были нажаты в один
   кадр записи, выравниваются по времени; курок 0/1 (цифровой) превращается
   в плавный наклон, если запись велась с такого устройства.

Скорость сортируется, «дыры» (пропуски ввода) подсчитываются и заливаются
интерполяцией — воспроизведение не должно получить разрыв.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dsp import lowpass, median_filter, resample, uniform_grid
from .frame import STICK_AXES, TRIGGER_AXES
from .session import Clip

STICK_AXIS_NAMES = tuple(a for pair in STICK_AXES.values() for a in pair)


def grid_is_regular(t: Sequence[float], tol: float = 0.02, runs: Optional[Sequence[int]] = None) -> bool:
    """Ровная ли сетка времени (шаг не гуляет сильнее ``tol`` от медианы).

    ``runs`` — начала независимых участков (кругов): внутри участка сетка
    оценивается отдельно, потому что между кругами есть пауза.
    """
    bounds = sorted({int(r) for r in (runs or []) if 0 < int(r) < len(t)})
    edges = [0] + bounds + [len(t)]
    for a, b in zip(edges, edges[1:]):
        part = list(t[a:b])
        if len(part) < 8:
            continue
        steps = sorted(part[i + 1] - part[i] for i in range(len(part) - 1))
        med = steps[len(steps) // 2]
        if med <= 0.0:
            return False
        if max(abs(x - med) for x in steps) > tol * med:
            return False
    return True


@dataclass
class ProcessConfig:
    """Параметры пост-обработки."""

    median: bool = True  # медиана по 3 точкам (выбросы)
    lowpass_hz: float = 14.0  # 0 = выключить сглаживание
    order: int = 4  # порядок ФНЧ «в одну сторону» (итого двойной)
    resample_hz: float = 120.0  # 0 = оставить сетку записи
    interp: str = "linear"  # linear | cubic
    deadzone: float = 0.0  # вырезание мёртвой зоны стиков
    trigger_expand: bool = True  # «0/1» курок -> плавный наклон
    trigger_ramp_ms: float = 35.0
    trim_idle: float = 0.0  # обрезать простой до N с (0 = выключено)
    idle_threshold: float = 0.08  # что считать «нейтралью» по стикам
    max_gap_ms: float = 60.0  # «дыра» в записи (для отчёта)
    smooth_buttons: bool = False  # сглаживать ли маску кнопок (обычно нет)


@dataclass
class ProcessResult:
    clip: Clip
    clip_raw: Clip
    stats: Dict[str, Any] = field(default_factory=dict)


def _find_gaps(t: Sequence[float], max_gap: float) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt > max_gap:
            out.append((i, dt))
    return out


def _trim_idle(clip: Clip, keep: float, threshold: float) -> Tuple[int, int, float, float]:
    """Найти границы «активности»: где стик/кнопки перестают быть нейтральными."""
    n = len(clip.t)
    active = []
    for i in range(n):
        mag = math.hypot(clip.lx[i], clip.ly[i]) + math.hypot(clip.rx[i], clip.ry[i])
        if mag > threshold or clip.btns[i] or clip.lt[i] > 0.05 or clip.rt[i] > 0.05:
            active.append(i)
    if not active:
        return 0, n - 1, 0.0, 0.0
    first, last = active[0], active[-1]
    start_t = max(0.0, clip.t[first] - keep)
    end_t = min(clip.duration, clip.t[last] + keep)
    return first, last, start_t, end_t


def process(clip: Clip, cfg: Optional[ProcessConfig] = None, *, source_name: str = "") -> ProcessResult:
    """Прогнать клип через пайплайн пост-обработки."""
    cfg = cfg or ProcessConfig()
    stats: Dict[str, Any] = {"input_samples": len(clip.t), "input_duration": clip.duration}

    if len(clip.t) < 3:
        return ProcessResult(clip=clip, clip_raw=clip, stats={"error": "слишком мало данных"})

    # 1) дыры во времени
    gaps = _find_gaps(clip.t, cfg.max_gap_ms / 1000.0)
    stats["gaps"] = len(gaps)
    stats["gap_total_s"] = round(sum(g[1] for g in gaps), 4)
    stats["gap_max_s"] = round(max((g[1] for g in gaps), default=0.0), 4)

    # 2) обрезка простоя
    if cfg.trim_idle > 0:
        first, last, start_t, end_t = _trim_idle(clip, cfg.trim_idle, cfg.idle_threshold)
        stats["trim"] = [round(start_t, 3), round(end_t, 3)]
        clip = clip.slice(start_t, end_t)
    else:
        stats["trim"] = None

    fs = clip.rate_hz or 250.0
    data: Dict[str, List[float]] = {ax: list(clip.axis(ax)) for ax in ("lx", "ly", "rx", "ry", "lt", "rt")}

    # 3) выбросы
    if cfg.median:
        for ax in STICK_AXIS_NAMES:
            data[ax] = median_filter(data[ax], 3)

    # 4) сглаживание
    if cfg.lowpass_hz and cfg.lowpass_hz > 0:
        for ax in STICK_AXIS_NAMES:
            data[ax] = lowpass(data[ax], cfg.lowpass_hz, fs, cfg.order)
        for ax in TRIGGER_AXES:
            data[ax] = lowpass(data[ax], max(cfg.lowpass_hz, 25.0), fs, order=2)

    # 5) «цифровые» курки -> плавный наклон
    if cfg.trigger_expand:
        ramp = cfg.trigger_ramp_ms / 1000.0
        for ax in TRIGGER_AXES:
            data[ax] = _ramp_triggers(data[ax], clip.t, ramp)

    # 6) мёртвая зона
    if cfg.deadzone > 0:
        lo, hi = cfg.deadzone, 1.0 - cfg.deadzone
        for ax in STICK_AXIS_NAMES:
            data[ax] = [
                0.0 if abs(v) <= lo else math.copysign(min(1.0, (abs(v) - lo) / max(hi, 1e-6)), v)
                for v in data[ax]
            ]

    # 7) ровная сетка
    do_resample = bool(cfg.resample_hz and cfg.resample_hz > 0 and abs(cfg.resample_hz - fs) > 1e-6)
    runs = clip.meta.get("run_starts") or None
    if do_resample and grid_is_regular(clip.t, runs=runs) and abs(cfg.resample_hz - fs) / max(fs, 1e-9) < 0.15:
        # сетка уже ровная и частоты близки: пересчёт не улучшит, а дробление
        # шага внесёт «ступеньки» (это видно по метрике дрожания)
        do_resample = False
        stats["resample_skipped"] = "сетка уже ровная"
    if do_resample:
        grid = uniform_grid(clip.duration, cfg.resample_hz)
        out = Clip(meta=dict(clip.meta))
        out.t = grid
        for ax in ("lx", "ly", "rx", "ry", "lt", "rt"):
            setattr(out, ax, [max(-1.0, min(1.0, v)) if ax in STICK_AXIS_NAMES else max(0.0, min(1.0, v))
                              for v in resample(data[ax], clip.t, grid, interp=cfg.interp)])
        # кнопки: «держать предыдущее состояние» — не пропустим короткое нажатие
        btns: List[int] = []
        i = 0
        for tt in grid:
            while i + 1 < len(clip.t) and clip.t[i + 1] <= tt:
                i += 1
            btns.append(int(clip.btns[min(i, len(clip.btns) - 1)]))
        out.btns = btns
        out.markers = list(clip.markers)
        stats["resampled_to"] = cfg.resample_hz
        stats["output_samples"] = len(out.t)
    else:
        out = Clip(meta=dict(clip.meta))
        out.t = list(clip.t)
        for ax in ("lx", "ly", "rx", "ry", "lt", "rt"):
            vals = data[ax]
            if ax in STICK_AXIS_NAMES:
                vals = [max(-1.0, min(1.0, v)) for v in vals]
            else:
                vals = [max(0.0, min(1.0, v)) for v in vals]
            setattr(out, ax, vals)
        out.btns = list(clip.btns)
        out.markers = list(clip.markers)
        stats["resampled_to"] = None
        stats["output_samples"] = len(out.t)

    stats["output_duration"] = round(out.duration, 4)
    stats["applied"] = {
        "median": cfg.median,
        "lowpass_hz": cfg.lowpass_hz,
        "resample_hz": cfg.resample_hz,
        "deadzone": cfg.deadzone,
        "trigger_expand": cfg.trigger_expand,
    }
    out.meta = dict(clip.meta)
    out.meta["processed"] = stats["applied"]
    if source_name:
        out.meta["source"] = source_name
    out.reindex()
    return ProcessResult(clip=out, clip_raw=clip, stats=stats)


def _ramp_triggers(values: Sequence[float], t: Sequence[float], ramp: float) -> List[float]:
    """Превратить «цифровой» курок (0/1) в наклон заданной длительности."""
    n = len(values)
    if n == 0 or ramp <= 0:
        return list(values)
    out = list(values)
    # найдём фронты 0->1 и 1->0
    for i in range(1, n):
        if out[i] > 0.5 >= out[i - 1]:  # начало нажатия: разгон вверх
            target = out[i]
            ramp_len = _count_within(t, i, ramp, forward=False)
            for k in range(max(0, i - ramp_len), i):
                f = (k - (i - ramp_len)) / max(ramp_len, 1)
                out[k] = max(out[k], target * f)
        elif out[i] <= 0.5 < out[i - 1]:  # отпускание: плавный спад
            ramp_len = _count_within(t, i, ramp, forward=True)
            start = out[i - 1]
            for k in range(i, min(n, i + ramp_len)):
                f = 1.0 - (k - i) / max(ramp_len, 1)
                out[k] = max(out[k], start * f)
    return out


def _count_within(t: Sequence[float], i: int, window: float, *, forward: bool) -> int:
    """Сколько сэмплов покрывает окно ``window`` секунд в нужную сторону."""
    n = len(t)
    count = 0
    if forward:
        j = i
        while j + 1 < n and (t[j + 1] - t[i]) <= window:
            j += 1
            count += 1
    else:
        j = i
        while j - 1 >= 0 and (t[i] - t[j - 1]) <= window:
            j -= 1
            count += 1
    return max(1, count)


def describe_process(stats: Dict[str, Any]) -> str:
    if stats.get("error"):
        return f"Обработка: {stats['error']}"
    lines = [
        f"Обработка: {stats['input_samples']} → {stats['output_samples']} сэмплов, "
        f"{stats['input_duration']:.2f} → {stats['output_duration']:.2f} с"
    ]
    a = stats.get("applied", {})
    bits = []
    if a.get("median"):
        bits.append("медиана 3")
    if a.get("lowpass_hz"):
        bits.append(f"ФНЧ {a['lowpass_hz']:.0f} Гц")
    if a.get("resample_hz"):
        bits.append(f"сетка {a['resample_hz']:.0f} Гц")
    if a.get("deadzone"):
        bits.append(f"мёртвая зона {a['deadzone']:.3f}")
    lines.append("Применено: " + (", ".join(bits) if bits else "ничего"))
    if stats.get("gaps"):
        lines.append(
            f"Пропусков ввода: {stats['gaps']} (суммарно {stats['gap_total_s']:.3f} с, "
            f"максимум {stats['gap_max_s'] * 1000:.0f} мс) — залиты интерполяцией"
        )
    if stats.get("trim"):
        lines.append(f"Обрезано до [{stats['trim'][0]:.2f} … {stats['trim'][1]:.2f}] с")
    return "\n".join(lines)


__all__ = ["ProcessConfig", "ProcessResult", "process", "describe_process"]

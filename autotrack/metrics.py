"""Метрики плавности: анализ «грибков» (стиков), кнопок и тайминга.

Зачем это нужно: оценить, насколько ровно скрипт ведёт машину по маршруту.
Метрики считаются в трёх точках:

* по **сырой записи** — что сделала рука и стоит ли чистить сигнал;
* по **плану вывода** — не «замылило» ли сглаживание траекторию;
* по **фактической выдаче** — доказательство, что в игру ушло то же самое.

Как считаются производные
-------------------------
Чистое численное дифференцирование превращает дрожание руки в мусор:
при 250 Гц шум 0.01 по стику даёт «рывок» порядка 10^5 ед/с³. Поэтому перед
дифференцированием сигнал фильтруется **нуль-фазовым** скользящим средним
(``deriv_cutoff_hz``, по умолчанию 25 Гц): реальное движение машины проходит
целиком, шум измерения — нет. Так же считаются «дрожание руки» (остаток
сигнала относительно фильтра 5 Гц) и «дрожание направления» (остаток угла).

Обозначения
-----------
* ``speed``          — |d(стик)/dt|, ед/с (полная шкала стика = 1.0);
* ``speed_cv``       — std/mean скорости: неравномерность хода (рывками);
* ``accel``         — ускорение стика, ед/с² (резкость);
* ``speed_jitter``  — колебания хода вокруг тренда, ед/с;
* ``noise_*``        — дрожание руки (остаток после фильтра 5 Гц);
* ``omega``          — угловая скорость направления, рад/с;
* ``alpha``          — угловое ускорение, рад/с² (дёрганье руля);
* ``direction_jitter_deg`` — СКО остатка угла, град (кривизна «пилой»);
* ``direction_flips``— устойчивые развороты направления (не считая дрожи);
* ``score``          — сводная плавность 0..100.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence

from .dsp import highpass, moving_average as _moving_average, window_for_cutoff
from .frame import STICK_AXES
from .session import Clip

try:  # необязательное ускорение (numpy есть почти всегда, но не обязателен)
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

EPS = 1e-9

#: Частота среза для оценки «дрожания» (остаток сигнала выше этой частоты).
NOISE_CUTOFF_HZ = 5.0
#: Частота среза для производных (скорость/ускорение/угловая скорость).
#: 12 Гц — выше любой реальной «руки», но режет шум перед двойным
#: дифференцированием (иначе ускорение/рывок измеряют только квантование).
DERIV_CUTOFF_HZ = 12.0

AXIS_NAMES = ("lx", "ly", "rx", "ry", "lt", "rt")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def pstdev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def percentiles(values: Sequence[float], ps: Sequence[float] = (50, 90, 95, 99, 100)) -> Dict[str, float]:
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    s = sorted(values)
    out: Dict[str, float] = {}
    for p in ps:
        idx = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
        out[f"p{p}"] = s[idx]
    return out


def sample_rate(t: Sequence[float], fallback: float = 250.0) -> float:
    if len(t) < 3:
        return fallback
    dts = [b - a for a, b in zip(t, t[1:]) if b > a]
    if not dts:
        return fallback
    m = statistics.median(dts)
    return 1.0 / m if m > 0 else fallback


def _win_for(fs: float, cutoff_hz: float, lo: int = 3, hi: int = 65) -> int:
    """Окно скользящего среднего, дающее срез примерно на ``cutoff_hz``."""
    return window_for_cutoff(fs, cutoff_hz, lo=lo, hi=hi)


def moving_average(values: Sequence[float], win: int) -> List[float]:
    """Нуль-фазовое сглаживание: скользящее среднее вперёд и назад."""
    return _moving_average(values, win)


def derivatives(values: Sequence[float], t: Sequence[float], win: int) -> List[List[float]]:
    """[x, x', x'', x'''] после нуль-фазового сглаживания (длины совпадают)."""
    n = len(values)
    if n < 2:
        return [list(values), [0.0] * n, [0.0] * n, [0.0] * n]
    x = moving_average(values, win)
    d1 = _diff_same_len(x, t)
    d2 = _diff_same_len(d1, t)
    d3 = _diff_same_len(d2, t)
    return [x, d1, d2, d3]


def _diff_same_len(vals: Sequence[float], t: Sequence[float]) -> List[float]:
    n = len(vals)
    if n < 2:
        return [0.0] * n
    if _np is not None:
        x = _np.asarray(vals, dtype=float)
        tt = _np.asarray(t, dtype=float)
        dt = _np.diff(tt)
        dt = _np.where(dt > 0, dt, 1e-6)
        d = _np.diff(x) / dt
        return [float(d[0])] + [float(v) for v in d]
    out = [0.0] * n
    last = 0.0
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        last = (vals[i] - vals[i - 1]) / dt if dt > 0 else last
        out[i] = last
    out[0] = out[1] if n > 1 else 0.0
    return out


def unwrap_angles(values: Sequence[float]) -> List[float]:
    """Развернуть углы, чтобы не было скачков через ±π (без разрывов)."""
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


def count_sustained_flips(omega: Sequence[float], threshold: float = 0.75, min_run: int = 3) -> int:
    """Сколько раз направление вращения устойчиво менялось (без дрожи)."""
    flips = 0
    sign = 0
    run = 0
    for w in omega:
        s = 1 if w > threshold else (-1 if w < -threshold else 0)
        if s == 0:
            continue
        if s == sign:
            run += 1
        else:
            if sign != 0 and run >= min_run:
                flips += 1
            sign = s
            run = 1
    return flips


def high_freq_ratio(values: Sequence[float], fs: float, cutoff: float = 12.0) -> float:
    """Доля энергии сигнала выше ``cutoff`` Гц (справочный показатель).

    Внимание: если в записи есть сильное медленное движение (поворот), оно
    «съедает» почти всю энергию и доля шума выглядит крошечной. Для оценки
    дрожания руки используйте ``noise_pct``.
    """
    n = len(values)
    if n < 32 or fs <= cutoff * 2:
        return 0.0
    if _np is None:
        return 0.0
    x = _np.asarray(values, dtype=float)
    x = x - x.mean()
    step = max(1, n // 2048)
    x = x[::step]
    fs_eff = fs / step
    spec = _np.abs(_np.fft.rfft(x * _np.hanning(x.size))) ** 2
    freqs = _np.fft.rfftfreq(x.size, d=1.0 / fs_eff)
    total = float(spec[1:].sum())
    high = float(spec[1:][freqs[1:] >= cutoff].sum())
    return high / total if total > 0 else 0.0


def noise_level(
    values: Sequence[float],
    fs: float,
    *,
    cutoff: float = NOISE_CUTOFF_HZ,
    scale: float = 1.0,
) -> Dict[str, float]:
    """Дрожание руки: высокочастотная составляющая сигнала (выше ``cutoff``).

    Используется нуль-фазовый ФВЧ Баттерворта (4-й порядок в каждую сторону,
    в сумме 8-й): медленный поворот машины (доли герца) почти не искажается и
    в шум не попадает, а дрожание руки 10–60 Гц проходит целиком.

    ``noise_pct`` — в процентах от полной шкалы стика (1.0 = край).
    """
    if len(values) < 16:
        return {"noise_rms": 0.0, "noise_pct": 0.0}
    res = highpass(values, cutoff, fs, order=4)
    # у любого нуль-фазового фильтра на краях есть небольшой переходный
    # процесс — он не имеет отношения к дрожанию руки, поэтому края отбрасываем
    edge = min(len(res) // 8, max(1, int(0.25 * fs)))
    if len(res) - 2 * edge >= 16:
        res = res[edge : len(res) - edge]
    nr = rms(res)
    ref = scale if scale > 0 else 1.0
    return {"noise_rms": nr, "noise_pct": 100.0 * nr / ref}


# ---------------------------------------------------------------------------
# Метрики канала и стика
# ---------------------------------------------------------------------------


def channel_metrics(values: Sequence[float], t: Sequence[float], label: str = "", fs: float = 0.0) -> Dict[str, Any]:
    n = len(values)
    if n < 3:
        return {"name": label, "samples": n}
    fs = fs or sample_rate(t)
    win = _win_for(fs, DERIV_CUTOFF_HZ)
    x, d1, d2, _d3 = derivatives(values, t, win)
    lo = noise_level(values, fs)
    rng = max(values) - min(values)
    return {
        "name": label,
        "samples": n,
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "std": pstdev(values),
        "range": rng,
        "speed_mean": mean([abs(v) for v in d1]),
        "speed_p95": percentiles([abs(v) for v in d1])["p95"],
        "accel_rms": rms(d2),
        "accel_p95": percentiles([abs(v) for v in d2])["p95"],
        "noise_rms": lo["noise_rms"],
        "noise_pct": lo["noise_pct"],
        "hf_ratio": high_freq_ratio(values, fs),
    }


def stick_metrics(
    xs: Sequence[float],
    ys: Sequence[float],
    t: Sequence[float],
    side: str = "L",
    *,
    deadzone: float = 0.05,
    deriv_cutoff: float = DERIV_CUTOFF_HZ,
    noise_cutoff: float = NOISE_CUTOFF_HZ,
) -> Dict[str, Any]:
    """Полный набор метрик одного стика, включая плавность направления."""
    n = min(len(xs), len(ys), len(t))
    xs, ys, t = list(xs[:n]), list(ys[:n]), list(t[:n])
    if n < 3:
        return {"name": side, "samples": n}
    fs = sample_rate(t)
    win = _win_for(fs, deriv_cutoff)
    win_slow = _win_for(fs, noise_cutoff, lo=3, hi=151)

    sx, dx, ddx, _ = derivatives(xs, t, win)
    sy, dy, ddy, _ = derivatives(ys, t, win)
    speed = [math.hypot(a, b) for a, b in zip(dx, dy)]
    accel = [math.hypot(a, b) for a, b in zip(ddx, ddy)]

    mags = [math.hypot(x, y) for x, y in zip(xs, ys)]
    angles = [math.atan2(y, x) for x, y in zip(xs, ys)]
    ang_unwrapped = unwrap_angles(angles)
    # ω, α — по развёрнутому углу; там, где стик в центре, направление
    # не имеет смысла, поэтому обнуляем (иначе получаем «шум центра»)
    ang_for_speed = [a if m > deadzone else 0.0 for a, m in zip(ang_unwrapped, mags)]
    _, omega, alpha, _ = derivatives(ang_for_speed, t, win)
    omega = [0.0 if m <= deadzone else w for w, m in zip(omega, mags)]
    alpha = [0.0 if m <= deadzone else a for a, m in zip(alpha, mags)]

    moving = [i for i, m in enumerate(mags) if m > deadzone]
    move_pct = 100.0 * len(moving) / n

    # дрожание: линейное (в % полной шкалы) и по направлению (в градусах)
    nx = noise_level(xs, fs, cutoff=noise_cutoff)
    ny = noise_level(ys, fs, cutoff=noise_cutoff)
    noise_rms = math.hypot(nx["noise_rms"], ny["noise_rms"])
    noise_pct = 100.0 * noise_rms  # шкала стика = 1.0

    ang_smooth = moving_average(ang_unwrapped, win_slow)
    ang_res = [
        (a - b) for a, b, m in zip(ang_unwrapped, ang_smooth, mags) if m > deadzone
    ]
    direction_jitter_deg = math.degrees(rms(ang_res))

    # неровность хода: колебания скорости вокруг её тренда (ед/с)
    speed_smooth = moving_average(speed, win_slow)
    speed_jitter = rms([a - b for a, b in zip(speed, speed_smooth)])

    speed_mean = mean(speed)
    rec: Dict[str, Any] = {
        "name": side,
        "samples": n,
        "duration": round(t[-1] - t[0], 4),
        "moving_pct": round(move_pct, 2),
        "mag_mean": mean(mags),
        "mag_max": max(mags),
        "mag_std": pstdev(mags),
        # линейная кинематика
        "speed_mean": speed_mean,
        "speed_p95": percentiles(speed)["p95"],
        "speed_max": max(speed),
        "speed_cv": pstdev(speed) / (speed_mean + EPS),
        "speed_jitter": speed_jitter,
        "accel_p95": percentiles(accel)["p95"],
        "accel_rms": rms(accel),
        # дрожание
        "noise_rms": noise_rms,
        "noise_pct": noise_pct,
        "noise_x_rms": nx["noise_rms"],
        "noise_y_rms": ny["noise_rms"],
        "hf_ratio": 0.5 * (high_freq_ratio(xs, fs) + high_freq_ratio(ys, fs)),
        # направление
        "angular_speed_mean": mean([abs(w) for w in omega]),
        "angular_speed_p95": percentiles([abs(w) for w in omega])["p95"],
        "angular_speed_max": max((abs(w) for w in omega), default=0.0),
        "angular_accel_p95": percentiles([abs(a) for a in alpha])["p95"],
        "angular_accel_rms": rms(alpha),
        "angular_jerk_rms": rms(_diff_same_len(alpha, t)),
        "direction_jitter_deg": direction_jitter_deg,
        "direction_flips": count_sustained_flips(omega),
        "returns_to_center": _center_returns(mags, deadzone),
        "circle_variability": _circle_variability(mags, moving),
    }
    rec["direction_flips_per_s"] = rec["direction_flips"] / max(t[-1] - t[0], EPS)
    rec["smoothness_score"] = smoothness_score(rec)
    return rec


def _center_returns(mags: Sequence[float], deadzone: float) -> int:
    count = 0
    inside = mags[0] <= deadzone
    for m in mags:
        if not inside and m <= deadzone:
            count += 1
            inside = True
        elif inside and m > deadzone:
            inside = False
    return count


def _circle_variability(mags: Sequence[float], moving: Sequence[int]) -> float:
    if len(moving) < 5:
        return 0.0
    vals = [mags[i] for i in moving]
    mx = max(vals) or 1.0
    return pstdev(vals) / mx


#: Пороги, после которых показатель считается «плохим».
#: noise_pct — в % полной шкалы стика (0.01 = 1 %),
#: speed_jitter — ед/с, α — рад/с², развороты — раз/с.
SCORE_THRESHOLDS = {
    "noise_pct": 1.5,
    "speed_cv": 1.5,
    "angular_accel_p95": 50.0,
    "flips_per_s": 4.0,
    "speed_jitter": 0.5,
}
SCORE_WEIGHTS = {
    "noise_pct": 0.30,
    "speed_cv": 0.15,
    "angular_accel_p95": 0.20,
    "flips_per_s": 0.15,
    "speed_jitter": 0.20,
}


def smoothness_score(m: Dict[str, Any]) -> float:
    """Сводная оценка плавности 0..100 (100 = идеально ровно)."""
    if not m or m.get("samples", 0) < 3:
        return 0.0
    penalty = 0.0
    for key, limit in SCORE_THRESHOLDS.items():
        value = float(m.get(key, 0.0))
        penalty += SCORE_WEIGHTS[key] * min(1.0, max(0.0, value) / limit)
    return round(max(0.0, 100.0 * (1.0 - penalty)), 1)


# ---------------------------------------------------------------------------
# Полный отчёт по клипу
# ---------------------------------------------------------------------------


def _merge_reports(parts, total_weight: float) -> Dict[str, Any]:
    """Слить отчёты нескольких участков: числа — среднее с весом по сэмплам."""
    def merge(a, b, wa, wb):
        out: Dict[str, Any] = {}
        for key in set(a) | set(b):
            va, vb = a.get(key), b.get(key)
            if key == "samples":
                out[key] = int(va or 0) + int(vb or 0)
            elif key == "duration":
                out[key] = round(float(va or 0.0) + float(vb or 0.0), 4)
            elif isinstance(va, dict) and isinstance(vb, dict):
                out[key] = merge(va, vb, wa, wb)
            elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                out[key] = (float(va) * wa + float(vb) * wb) / max(wa + wb, 1e-9)
            else:
                out[key] = va if va is not None else vb
        return out

    acc, weight = parts[0][0], float(parts[0][1])
    for rep, w in parts[1:]:
        acc, weight = merge(acc, rep, weight, float(w)), weight + float(w)
    return _round_report(acc)


def _round_report(rep: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
    """Округлить числа отчёта (после усреднения) как в обычном analyze()."""
    out: Dict[str, Any] = {}
    for key, val in rep.items():
        if isinstance(val, dict):
            out[key] = _round_report(val, depth + 1)
        elif isinstance(val, float):
            out[key] = round(val, 4)
        else:
            out[key] = val
    return out


def analyze_runs(clip: Clip, runs, **kw) -> Dict[str, Any]:
    """Отчёт по клипу с разрезкой на независимые участки (круги маршрута).

    Стык двух кругов — это не дрожание стика: игрок (и игра) видит разрыв,
    которого в самом маршруте нет. Поэтому каждый круг считается отдельно,
    а числа усредняются с весом по числу сэмплов.
    """
    bounds = sorted({int(r) for r in (runs or []) if 0 < int(r) < len(clip.t)})
    if not bounds:
        return analyze(clip, **kw)
    edges = [0] + bounds + [len(clip.t)]
    parts = []
    for a, b in zip(edges, edges[1:]):
        if b - a < 8:
            continue
        sub = Clip(meta={k: v for k, v in clip.meta.items() if k != "run_starts"})
        sub.meta["run"] = len(parts) + 1
        for i in range(a, b):
            sub.append_raw(clip.t[i] - clip.t[a], clip.frame(i))
        parts.append((sub, b - a))
    if len(parts) < 2:
        return analyze(clip, _split=False, **kw)
    reports = [(analyze(sub, _split=False, **kw), float(w)) for sub, w in parts]
    merged = _merge_reports(reports, float(sum(w for _, w in parts)))
    merged["runs"] = len(parts)
    merged["clip"] = clip.name
    return merged


def analyze(
    clip: Clip,
    *,
    deadzone: float = 0.05,
    buttons: bool = True,
    deriv_cutoff: float = DERIV_CUTOFF_HZ,
    _split: bool = True,
) -> Dict[str, Any]:
    """Полный отчёт: оси, стики, кнопки, сводка.

    Если в клипе сохранены границы кругов (`meta["run_starts"]`), каждый круг
    считается отдельно: разрыв между кругами — это не дрожание стика.
    """
    if len(clip.t) < 3:
        return {"samples": len(clip.t), "error": "слишком мало данных"}
    if _split:
        starts = clip.meta.get("run_starts") or []
        if isinstance(starts, (list, tuple)) and len(starts) > 1:
            return analyze_runs(
                clip,
                starts,
                deadzone=deadzone,
                buttons=buttons,
                deriv_cutoff=deriv_cutoff,
            )
    fs = sample_rate(clip.t)
    report: Dict[str, Any] = {
        "clip": clip.name,
        "samples": len(clip.t),
        "duration": round(clip.duration, 4),
        "rate_hz": round(fs, 2),
        "axes": {},
        "sticks": {},
    }
    for ax in AXIS_NAMES:
        report["axes"][ax] = channel_metrics(clip.axis(ax), clip.t, ax, fs)
    for side, (ax, ay) in STICK_AXES.items():
        report["sticks"][side] = stick_metrics(
            clip.axis(ax), clip.axis(ay), clip.t, side=side, deadzone=deadzone, deriv_cutoff=deriv_cutoff
        )
    if buttons:
        runs = clip.button_runs()
        by_button: Dict[str, Dict[str, Any]] = {}
        for r in runs:
            b = by_button.setdefault(
                r["button"], {"count": 0, "total": 0.0, "min": float("inf"), "max": 0.0}
            )
            b["count"] += 1
            b["total"] += r["duration"]
            b["min"] = min(b["min"], r["duration"])
            b["max"] = max(b["max"], r["duration"])
        for name, b in by_button.items():
            b["total"] = round(b["total"], 3)
            b["min"] = round(0.0 if b["min"] == float("inf") else b["min"], 3)
            b["max"] = round(b["max"], 3)
            b["per_minute"] = round(b["count"] / max(clip.duration, 1e-6) * 60.0, 1)
        report["buttons"] = by_button
        report["short_presses"] = [r for r in runs if r["duration"] < 0.04]
    sticks = list(report["sticks"].values())
    if sticks:
        weights = [max(s.get("moving_pct", 0.0), 5.0) for s in sticks]
        report["smoothness_score"] = round(
            sum(s.get("smoothness_score", 0.0) * w for s, w in zip(sticks, weights)) / sum(weights), 1
        )
        report["issues"] = collect_issues(report)
    return report


def collect_issues(report: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    for side, s in report.get("sticks", {}).items():
        if s.get("samples", 0) < 3:
            continue
        if s.get("noise_pct", 0) > SCORE_THRESHOLDS["noise_pct"]:
            issues.append(
                f"стик {side}: дрожание руки {s['noise_pct']:.1f}% — включите сглаживание (--min-cutoff ниже / --beta выше)"
            )
        if s.get("speed_cv", 0) > SCORE_THRESHOLDS["speed_cv"]:
            issues.append(f"стик {side}: неравномерная скорость (CV={s['speed_cv']:.2f}) — рывками")
        if s.get("speed_jitter", 0) > SCORE_THRESHOLDS["speed_jitter"]:
            issues.append(f"стик {side}: колебания хода {s['speed_jitter']:.2f} ед/с — газ/руль не удержаны")
        if s.get("direction_flips_per_s", 0) > SCORE_THRESHOLDS["flips_per_s"]:
            issues.append(
                f"стик {side}: направление разворачивается {s['direction_flips_per_s']:.1f} раз/с — «пилит»"
            )
        if s.get("angular_accel_p95", 0) > SCORE_THRESHOLDS["angular_accel_p95"]:
            issues.append(f"стик {side}: дёрганое вращение, угловое ускорение p95={s['angular_accel_p95']:.1f} рад/с²")
        if s.get("direction_jitter_deg", 0) > 4.0:
            issues.append(f"стик {side}: дрожание направления {s['direction_jitter_deg']:.1f}°")
    if report.get("short_presses"):
        issues.append(f"кнопки: нажатий короче 40 мс — {len(report['short_presses'])}")
    return issues


def verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    """Качественный вердикт по отчёту."""
    score = report.get("smoothness_score", 0.0)
    issues = report.get("issues", [])
    if report.get("error"):
        return {"level": "нет данных", "score": 0.0, "text": report["error"]}
    if score >= 85:
        level = "отлично"
    elif score >= 70:
        level = "хорошо"
    elif score >= 50:
        level = "средне"
    else:
        level = "плохо"
    return {"level": level, "score": score, "issues": issues}


# ---------------------------------------------------------------------------
# Печать
# ---------------------------------------------------------------------------


def format_report(report: Dict[str, Any], *, title: str = "ОТЧЁТ О ПЛАВНОСТИ", issues_limit: int = 8) -> str:
    if report.get("error"):
        return f"{title}: {report['error']}"
    v = verdict(report)
    w = 78
    lines = ["=" * w, f" {title}", "=" * w]
    lines.append(
        f" Клип: {report.get('clip')} | {report.get('duration')} с | {report.get('samples')} сэмплов "
        f"| ~{report.get('rate_hz')} Гц"
    )
    lines.append(f" ПЛАВНОСТЬ: {report.get('smoothness_score', 0)}/100 ({v['level']})")
    lines.append("-" * w)
    lines.append(
        f" {'стик':<5}{'ход,%':>7}{'скор':>6}{'p95':>7}{'CV':>6}{'рывки':>7}"
        f"{'дрожь,%':>9}{'ω':>6}{'α p95':>8}{'развор':>7}{'угол°':>7}{'оценка':>8}"
    )
    for side in ("L", "R"):
        s = report.get("sticks", {}).get(side)
        if not s or s.get("samples", 0) < 3:
            continue
        lines.append(
            f" {side:<5}{s['moving_pct']:>7.1f}{s['speed_mean']:>6.2f}{s['speed_p95']:>7.2f}"
            f"{s['speed_cv']:>6.2f}{s['speed_jitter']:>7.2f}{s['noise_pct']:>9.2f}"
            f"{s['angular_speed_mean']:>6.2f}{s['angular_accel_p95']:>8.1f}"
            f"{s['direction_flips_per_s']:>7.2f}{s['direction_jitter_deg']:>7.1f}"
            f"{s['smoothness_score']:>8.1f}"
        )
    lines.append("-" * w)
    lines.append(
        " обозначения: скор=ед/с, p95=95-й процентиль скорости, CV=неравномерность хода,"
    )
    lines.append(
        " рывки=колебания хода ед/с, дрожь=% полной шкалы (дрожание руки), ω=рад/с, "
        "α p95=рад/с², развор=раз/с, угол°=дрожание направления"
    )
    for side in ("L", "R"):
        s = report.get("sticks", {}).get(side)
        if not s or s.get("samples", 0) < 3:
            continue
        lines.append(
            f" Стик {side}: возвратов в центр {s['returns_to_center']}, макс.амплитуда {s['mag_max']:.2f}, "
            f"угловая скорость p95 {s['angular_speed_p95']:.2f} рад/с, "
            f"ускорение p95 {s['accel_p95']:.1f} ед/с², дрожание {s['noise_rms']:.4f} ед"
        )
    btns = report.get("buttons") or {}
    if btns:
        top = sorted(btns.items(), key=lambda kv: -kv[1]["count"])[:8]
        lines.append(
            " Кнопки: "
            + ", ".join(
                f"{n}×{d['count']} (ср {d['total'] / max(d['count'], 1) * 1000:.0f} мс, макс {d['max'] * 1000:.0f} мс)"
                for n, d in top
            )
        )
    issues = report.get("issues") or []
    if issues:
        lines.append("-" * w)
        lines.append(" Замечания:")
        for issue in issues[:issues_limit]:
            lines.append(f"   • {issue}")
        if len(issues) > issues_limit:
            lines.append(f"   … и ещё {len(issues) - issues_limit}")
    lines.append("=" * w)
    return "\n".join(lines)


def compare_reports(before: Dict[str, Any], after: Dict[str, Any], *, title: str = "ДО/ПОСЛЕ") -> str:
    lines = ["", f"СРАВНЕНИЕ ({title})", "-" * 78]
    lines.append(
        f" {'стик':<5}{'дрожь до':>10}{'после':>8}{'рывки до':>10}{'после':>8}"
        f"{'угол° до':>10}{'после':>8}{'α p95':>16}{'оценка':>18}"
    )
    for side in ("L", "R"):
        b, a = before.get("sticks", {}).get(side), after.get("sticks", {}).get(side)
        if not b or not a or b.get("samples", 0) < 3:
            continue
        lines.append(
            f" {side:<5}{b['noise_pct']:>10.2f}{a['noise_pct']:>8.2f}"
            f"{b['speed_jitter']:>10.2f}{a['speed_jitter']:>8.2f}"
            f"{b['direction_jitter_deg']:>10.1f}{a['direction_jitter_deg']:>8.1f}"
            f"{b['angular_accel_p95']:>8.1f} →{a['angular_accel_p95']:>6.1f}"
            f"{b['smoothness_score']:>7.1f} →{a['smoothness_score']:>6.1f}"
        )
    lines.append("-" * 78)
    lines.append(
        f" Итоговая плавность: {float(before.get('smoothness_score', 0)):.1f}"
        f" → {float(after.get('smoothness_score', 0)):.1f}"
    )
    return "\n".join(lines)


def playback_verification(planned: Clip, emitted: Clip, *, deadzone: float = 0.05) -> Dict[str, Any]:
    """Сверка «что планировали → что реально ушло в геймпад»."""
    n = min(len(planned.t), len(emitted.t))
    if n < 2:
        return {"error": "недостаточно данных для проверки"}
    err_max: Dict[str, float] = {}
    err_rms: Dict[str, float] = {}
    for ax in ("lx", "ly", "rx", "ry", "lt", "rt"):
        a, b = planned.axis(ax)[:n], emitted.axis(ax)[:n]
        diffs = [abs(x - y) for x, y in zip(a, b)]
        err_max[ax] = max(diffs)
        err_rms[ax] = rms(diffs)
    planned_runs = planned.button_runs()
    emitted_runs = emitted.button_runs()
    missed = 0
    for r in planned_runs:
        if not any(
            x["button"] == r["button"] and abs(x["start"] - r["start"]) < 0.05 for x in emitted_runs
        ):
            missed += 1
    return {
        "compared_samples": n,
        "axis_error_max": {k: round(v, 5) for k, v in err_max.items()},
        "axis_error_rms": {k: round(v, 5) for k, v in err_rms.items()},
        "buttons_planned": len(planned_runs),
        "buttons_missed": missed,
    }


__all__ = [
    "analyze",
    "stick_metrics",
    "channel_metrics",
    "smoothness_score",
    "format_report",
    "compare_reports",
    "verdict",
    "playback_verification",
    "sample_rate",
    "percentiles",
    "moving_average",
    "noise_level",
    "count_sustained_flips",
    "unwrap_angles",
]

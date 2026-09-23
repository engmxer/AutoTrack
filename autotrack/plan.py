"""План вывода: превращение записанного клипа в точную последовательность кадров.

Что здесь происходит (это и есть ответ на «воспринимал джойстик и повторял его
действия без перебоев и лагов»):

1. клип приводится к внутренней частоте вывода (``rate_hz``, по умолчанию
   250 Гц — каждые 4 мс, вдвое чаще кадра игры в 120 FPS);
2. незначащие микро-движения стика у центра давятся (``deadzone``), чтобы не
   «сбивать» захват направления в игре;
3. каждая посылка состояния интерполируется между записанными точками
   (линейно или кубически) — игра не видит «ступенек»;
4. короткие нажатия кнопок растягиваются до ``min_press_ticks``: иначе игра,
   рисующая 60 кадров/с, может просто не увидеть нажатие длиной 1 кадр
   виртуального геймпада;
5. на выход идёт монотонно возрастающий индекс тика — никаких «прыжков во
   времени», отставание не накапливается (см. ``DeadlineLoop``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

from .frame import AXES, PadState, clamp
from .session import Clip

#: Порог, выше которого курок считается нажатым (шмитт).
TRIGGER_THRESHOLD = 0.04


@dataclass
class PlanConfig:
    """Настройки подготовки плана вывода."""

    rate_hz: float = 250.0
    speed: float = 1.0
    loops: int = 0  # 0 = бесконечно
    loop_gap: float = 0.15  # пауза между кругами, с
    deadzone: float = 0.0  # вырезание «нуля» стика (обычно не нужно: чистим при записи)
    stick_deadzone_scale: bool = True  # сжимать диапазон после вырезания мёртвой зоны
    interp: str = "linear"  # linear | cubic | hold
    game_fps: float = 60.0
    min_press_scale: float = 1.5
    tail_hold_ms: float = 30.0  # держать последний кадр в конце (для читаемости)
    lead_in_ms: float = 0.0  # сколько держать нейтраль перед стартом
    trim_start: float = 0.0
    trim_end: float = 0.0
    finish_neutral: bool = True  # отпустить всё в самом конце
    entry_ramp_ms: float = 0.0  # плавный вход из нейтрали в начале круга
    exit_ramp_ms: float = 0.0  # плавный выход в нейтраль в конце круга

    @property
    def period(self) -> float:
        return 1.0 / self.rate_hz


@dataclass
class Plan:
    """Готовый к воспроизведению маршрут (в тиках виртуального геймпада)."""

    config: PlanConfig
    ticks: int  # тиков в одном круге
    times: List[int]  # тик для каждого сэмпла (монотонно, 0..ticks)
    states: List[PadState]  # состояние на сэмпл
    sample_rate: float  # частота сэмплов (обычно = rate_hz)
    markers: List[Tuple[int, str]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # -- сглаживание --------------------------------------------------------
    def smooth_zero_phase(
        self,
        cutoff_hz: float = 10.0,
        *,
        order: int = 2,
        deadzone: float = 0.02,
    ) -> "Plan":
        """Сгладить маршрут нуль-фазовым фильтром (без задержки отклика).

        Работает по уже готовому плану: правки вносятся в сами тики, поэтому
        сэмплер и телеметрия видят уже сглаженный маршрут, а реплика не
        сдвигается во времени.
        """
        from .filters import zero_phase_stick

        fs = self.config.rate_hz
        if cutoff_hz <= 0.0 or len(self.states) < 8:
            return self
        lx, ly = zero_phase_stick(
            [st.lx for st in self.states],
            [st.ly for st in self.states],
            fs,
            cutoff_hz,
            order=order,
            deadzone=deadzone,
        )
        rx, ry = zero_phase_stick(
            [st.rx for st in self.states],
            [st.ry for st in self.states],
            fs,
            cutoff_hz,
            order=order,
            deadzone=deadzone,
        )
        # PadState — неизменяемый NamedTuple, поэтому собираем новый список
        self.states = [
            st._replace(lx=lx[i], ly=ly[i], rx=rx[i], ry=ry[i])
            for i, st in enumerate(self.states)
        ]
        self.meta["smooth_hz"] = cutoff_hz
        return self

    # -- свойства -----------------------------------------------------------
    @property
    def duration(self) -> float:
        return self.ticks / self.config.rate_hz

    def gap_ticks(self) -> int:
        return max(0, int(round(self.config.loop_gap * self.config.rate_hz)))

    @property
    def lead_ticks(self) -> int:
        return max(0, int(round(self.config.lead_in_ms / 1000.0 * self.config.rate_hz)))

    @property
    def tail_ticks(self) -> int:
        return max(0, int(round(self.config.tail_hold_ms / 1000.0 * self.config.rate_hz)))

    def total_ticks(self) -> Optional[int]:
        """Сколько всего тиков продлится сессия (для конечного числа кругов)."""
        if self.config.loops <= 0:
            return None
        loops = self.config.loops
        return (
            self.lead_ticks
            + loops * self.ticks
            + max(0, loops - 1) * self.gap_ticks()
            + (self.tail_ticks if self.config.finish_neutral else 0)
        )

    def iter_states(self) -> Iterator[Tuple[int, int, PadState]]:
        """Генератор ``(глобальный тик, номер круга, состояние)`` на всю сессию."""
        loops = self.config.loops if self.config.loops > 0 else None
        cycle = 0
        lead = self.lead_ticks
        tail = self.tail_ticks
        base = 0
        for k in range(lead):
            yield k, 0, PadState()
        base = lead
        while loops is None or cycle < loops:
            sampler = Sampler(self)
            for local in range(self.ticks):
                yield base + local, cycle, sampler(local)
            cycle += 1
            if loops is not None and cycle >= loops:
                break
            gap = self.gap_ticks()
            for g in range(gap):
                yield base + self.ticks + g, cycle - 1, PadState()
            base += self.ticks + gap
        if self.config.finish_neutral:
            for x in range(tail):
                yield base + self.ticks + x, cycle - 1, PadState()

    def state_at(self, local_tick: int) -> PadState:
        """Состояние на локальном тике внутри одного круга (без кэша)."""
        return Sampler(self)(local_tick)

    def to_clip(self, *, loops: int = 1) -> Clip:
        """Развернуть план обратно в клип (для отчётов/сохранения)."""
        cfg = self.config
        clip = Clip(meta=dict(self.meta, name="plan"))
        old_loops = cfg.loops
        try:
            cfg.loops = loops
            for k, _cycle, state in self.iter_states():
                clip.append(k / cfg.rate_hz, state)
        finally:
            cfg.loops = old_loops
        return clip


class Sampler:
    """Интерполятор: локальный тик -> состояние геймпада.

    Индекс по сэмплам увеличивается монотонно (указатель), поэтому вызов
    стоит несколько микросекунд — цикл 250 Гц тратит на это <1 % бюджета.
    """

    __slots__ = ("plan", "times", "states", "n", "interp", "i", "_t0", "_t1", "_s0", "_s1")

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.times = plan.times
        self.states = plan.states
        self.n = len(plan.times)
        self.interp = plan.config.interp
        self.reset()

    def reset(self) -> None:
        self.i = 0
        self._t0 = self._t1 = 0
        self._s0 = self._s1 = PadState()

    # -- интерполяция -------------------------------------------------------
    def _pick(self, k: int) -> None:
        times = self.times
        i = self.i
        n = self.n
        if n == 0:
            self._s0 = self._s1 = PadState()
            return
        while i + 1 < n and times[i + 1] <= k:
            i += 1
        if i >= n - 1:
            i = n - 1
        self.i = i
        self._s0 = self._s1 = self.states[i]
        if i + 1 < n:
            self._t0, self._t1 = times[i], times[i + 1]
            self._s1 = self.states[i + 1]

    def __call__(self, k: int) -> PadState:
        if self.n == 0:
            return PadState()
        self._pick(k)
        a, b = self._s0, self._s1
        span = self._t1 - self._t0
        if span <= 0 or self.interp == "hold":
            frac = 0.0
        else:
            frac = (k - self._t0) / span
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
        if frac == 0.0:
            return a
        if self.interp == "cubic":
            # Catmull-Rom по 4 точкам вокруг (если есть соседи)
            i = self.i
            p0 = self.states[max(0, i - 1)]
            p1 = a
            p2 = b
            p3 = self.states[min(self.n - 1, i + 2)]
            return _catmull_rom(p0, p1, p2, p3, frac)
        # линейная интерполяция
        return PadState(
            lx=a.lx + (b.lx - a.lx) * frac,
            ly=a.ly + (b.ly - a.ly) * frac,
            rx=a.rx + (b.rx - a.rx) * frac,
            ry=a.ry + (b.ry - a.ry) * frac,
            lt=a.lt + (b.lt - a.lt) * frac,
            rt=a.rt + (b.rt - a.rt) * frac,
            btns=a.btns,
        )


def _catmull_rom(p0: PadState, p1: PadState, p2: PadState, p3: PadState, f: float) -> PadState:
    f2 = f * f
    f3 = f2 * f

    def interp(a: float, b: float, c: float, d: float) -> float:
        # классический Catmull-Rom (равномерный)
        v = 0.5 * (
            (2.0 * b)
            + (-a + c) * f
            + (2.0 * a - 5.0 * b + 4.0 * c - d) * f2
            + (-a + 3.0 * b - 3.0 * c + d) * f3
        )
        return v

    # кнопки — «держать предыдущее», как в игре; триггеры — кубически с зажимом
    return PadState(
        lx=clamp(interp(p0.lx, p1.lx, p2.lx, p3.lx)),
        ly=clamp(interp(p0.ly, p1.ly, p2.ly, p3.ly)),
        rx=clamp(interp(p0.rx, p1.rx, p2.rx, p3.rx)),
        ry=clamp(interp(p0.ry, p1.ry, p2.ry, p3.ry)),
        lt=clamp(interp(p0.lt, p1.lt, p2.lt, p3.lt), 0.0, 1.0),
        rt=clamp(interp(p0.rt, p1.rt, p2.rt, p3.rt), 0.0, 1.0),
        btns=p1.btns,
    )


def _ease_in(f: float) -> float:
    """Косинусная огибающая 0→1: производная на концах нулевая (нет «клевания»)."""
    return 0.5 - 0.5 * math.cos(math.pi * min(max(f, 0.0), 1.0))


def _ease_out(f: float) -> float:
    """Косинусная огибающая 1→0."""
    return 1.0 - _ease_in(f)


def _fade(state: PadState, f: float) -> PadState:
    """Масштабировать оси состояния (плавный вход/выход). Кнопки не трогаем."""
    if f >= 1.0:
        return state
    return PadState(
        lx=state.lx * f,
        ly=state.ly * f,
        rx=state.rx * f,
        ry=state.ry * f,
        lt=state.lt * f,
        rt=state.rt * f,
        btns=state.btns,
    )


def _resample_axes(
    t: Sequence[float],
    values: Sequence[float],
    target_t: Sequence[float],
    *,
    interp: str = "linear",
) -> List[float]:
    """Пересэмплировать ряд на новую сетку времени (без numpy)."""
    out: List[float] = []
    n = len(t)
    if n == 0:
        return [0.0] * len(target_t)
    i = 0
    for tt in target_t:
        while i + 1 < n and t[i + 1] <= tt:
            i += 1
        j = min(i, n - 1)
        if j + 1 >= n or t[j + 1] <= t[j]:
            out.append(values[j])
            continue
        span = t[j + 1] - t[j]
        f = 0.0 if span <= 0 else (tt - t[j]) / span
        f = 0.0 if f < 0 else (1.0 if f > 1 else f)
        if interp == "hold" or f == 0.0:
            out.append(values[j])
        elif interp == "cubic":
            a = values[max(0, j - 1)]
            b = values[j]
            c = values[j + 1]
            d = values[min(n - 1, j + 2)]
            out.append(_cr(a, b, c, d, f))
        else:
            out.append(values[j] + (values[j + 1] - values[j]) * f)
    return out


def _cr(a: float, b: float, c: float, d: float, f: float) -> float:
    f2 = f * f
    f3 = f2 * f
    return 0.5 * (
        (2.0 * b) + (-a + c) * f + (2.0 * a - 5.0 * b + 4.0 * c - d) * f2 + (-a + 3.0 * b - 3.0 * c + d) * f3
    )


def _apply_deadzone(series: List[float], deadzone: float, scale: bool, *, smooth: bool = True) -> List[float]:
    """Мёртвая зона с **плавным** переходом (резкий излом добавляет дрожание)."""
    lo = deadzone
    if lo <= 0.0:
        return series
    hi = 1.0 - lo
    band = lo  # ширина плавного перехода
    out: List[float] = []
    for v in series:
        a = abs(v)
        if a <= lo:
            out.append(0.0)
        elif smooth and a < lo + band and hi > 1e-6:
            # smoothstep от 0 до линейного продолжения
            f = (a - lo) / band
            f = f * f * (3.0 - 2.0 * f)
            out.append(math.copysign(f * (a - lo) / hi, v))
        elif scale and hi > 1e-6:
            out.append(math.copysign((a - lo) / hi, v))
        else:
            out.append(v)
    return out


def _merge_button_runs(masks: List[int], min_ticks: int, tick_at) -> List[int]:
    """Гарантировать минимальную длительность нажатия (в тиках).

    Игра, рисующая 60 кадров/с, может не увидеть нажатие длиной в 1–2 тика
    виртуального геймпада (4–8 мс). Поэтому короткие нажатия «дотягиваются»
    назад (если рядом с началом записи) или вперёд до ``min_ticks``.
    """
    n = len(masks)
    if n == 0 or min_ticks <= 1:
        return masks
    from .frame import BUTTON_BITS

    for _name, bitno in BUTTON_BITS.items():
        b = 1 << bitno
        i = 0
        while i < n:
            if not (masks[i] & b):
                i += 1
                continue
            start = i
            while i < n and (masks[i] & b):
                i += 1
            end = i  # нажатие занимает тики [tick_at(start), tick_at(end))
            t_start, t_end = tick_at(start), tick_at(end)
            if t_end - t_start >= min_ticks:
                continue
            keep_until = t_start + min_ticks
            j = end
            while j < n and tick_at(j) < keep_until:
                masks[j] |= b
                j += 1
            # если упёрлись в конец записи — начинаем нажатие раньше
            k = start - 1
            while k >= 0 and (t_end - tick_at(k)) < min_ticks:
                masks[k] |= b
                k -= 1
            # важно: продолжать поиск ПОСЛЕ растянутого участка, иначе
            # «хвост» растяжки будет снова найден и нажатие разрастётся
            i = max(i, j)
    return masks


def _merge_trigger_pulses(vals: List[float], min_ticks: int, tick_at) -> List[float]:
    """Короткие импульсы курков (например ``RT`` на 4 мс) растянуть до ``min_ticks``."""
    n = len(vals)
    if n == 0 or min_ticks <= 1:
        return list(vals)
    out = list(vals)
    i = 0
    while i < n:
        if out[i] <= TRIGGER_THRESHOLD:
            i += 1
            continue
        start = i
        peak = out[i]
        while i < n and out[i] > TRIGGER_THRESHOLD:
            peak = max(peak, out[i])
            i += 1
        end = i
        if tick_at(end) - tick_at(start) >= min_ticks:
            continue
        keep_until = tick_at(start) + min_ticks
        j = end
        while j < n and tick_at(j) < keep_until:
            out[j] = max(peak, out[j])
            j += 1
        k = start - 1
        while k >= 0 and (tick_at(end) - tick_at(k)) < min_ticks:
            out[k] = max(peak, out[k])
            k -= 1
        i = max(i, j)
    return out


def build_plan(clip: Clip, config: Optional[PlanConfig] = None) -> Plan:
    """Собрать план вывода из клипа."""
    cfg = config or PlanConfig()
    cfg.rate_hz = float(cfg.rate_hz)
    cfg.speed = float(cfg.speed) or 1.0
    scale = cfg.speed / cfg.rate_hz
    if cfg.speed:
        src = clip
        if cfg.trim_start or cfg.trim_end:
            end = clip.duration - max(0.0, cfg.trim_end)
            src = clip.slice(max(0.0, cfg.trim_start), max(0.0, end))
        ticks = max(1, int(round(src.duration / scale)))
        target_t = [i * scale for i in range(ticks + 1)]
        interp = cfg.interp

        axes = {}
        for name in AXES:
            vals = _resample_axes(src.t, src.axis(name), target_t, interp=interp)
            if name in ("lx", "ly", "rx", "ry"):
                vals = _apply_deadzone(vals, cfg.deadzone, cfg.stick_deadzone_scale)
            else:
                vals = [max(0.0, min(1.0, v)) for v in vals]
            axes[name] = vals

        # кнопки: берём состояние, действовавшее на момент сэмпла
        btn_src = [int(b) for b in src.btns]
        btns: List[int] = []
        i = 0
        for tt in target_t:
            while i + 1 < len(src.t) and src.t[i + 1] <= tt:
                i += 1
            btns.append(btn_src[min(i, len(btn_src) - 1)] if btn_src else 0)

        min_press = max(1, int(math.ceil(cfg.rate_hz / max(1.0, cfg.game_fps) * cfg.min_press_scale)))
        min_press = int(min(min_press, max(1, ticks)))

        # Растяжка коротких нажатий. Сетка уже ровная и в тиках, поэтому
        # «время» сэмпла — его индекс: tick_at(k) == k.
        def tick_at(k: int) -> int:
            return k

        btns = _merge_button_runs(btns, min_press, tick_at)
        for name in ("lt", "rt"):
            axes[name] = _merge_trigger_pulses(axes[name], min_press, tick_at)

        states = [
            PadState(
                lx=axes["lx"][k], ly=axes["ly"][k], rx=axes["rx"][k], ry=axes["ry"][k],
                lt=axes["lt"][k], rt=axes["rt"][k], btns=int(btns[k]),
            )
            for k in range(ticks + 1)
        ]

        # Плавный вход/выход: без «удара» стиком, когда маршрут начинается сразу
        # с отклонённого положения, и без резкого отпускания в конце круга.
        ramp_in = max(0, int(round(cfg.entry_ramp_ms / 1000.0 * cfg.rate_hz)))
        ramp_out = max(0, int(round(cfg.exit_ramp_ms / 1000.0 * cfg.rate_hz)))
        n_states = len(states)
        if ramp_in > 1:
            span = min(ramp_in, n_states)
            for i in range(span):
                states[i] = _fade(states[i], _ease_in(i / max(span - 1, 1)))
        if ramp_out > 1:
            span = min(ramp_out, n_states)
            first = n_states - span
            for k in range(span):
                u = k / max(span - 1, 1)
                states[first + k] = _fade(states[first + k], _ease_out(u))

        markers = [
            (max(0, min(ticks, int(round((mt - cfg.trim_start) / scale)))), lbl)
            for mt, lbl in src.markers
        ]
        plan = Plan(
            config=cfg,
            ticks=ticks,
            times=list(range(ticks + 1)),
            states=states,
            sample_rate=cfg.rate_hz,
            markers=sorted(markers),
            meta=dict(src.meta, source=src.name),
        )
        plan.meta["min_press_ticks"] = min_press
        plan.meta["ticks"] = ticks
        return plan
    raise ValueError("speed должен быть больше нуля")


def describe_plan(plan: Plan) -> str:
    cfg = plan.config
    lines = [
        f"План: {plan.ticks} тиков по {cfg.period * 1000:.2f} мс "
        f"({plan.duration:.2f} с на круг, частота вывода {cfg.rate_hz:.0f} Гц)",
    ]
    if cfg.loops > 0:
        lines.append(f"Кругов: {cfg.loops} (пауза между кругами {cfg.loop_gap * 1000:.0f} мс)")
    else:
        lines.append("Кругов: бесконечно (остановка — ESC/F12 или Ctrl+C в консоли)")
    lines.append(
        f"Интерполяция: {cfg.interp}, мёртвая зона стиков: {cfg.deadzone:.3f}, "
        f"минимальное нажатие кнопки: {plan.meta.get('min_press_ticks')} тиков "
        f"({plan.meta.get('min_press_ticks', 0) * cfg.period * 1000:.1f} мс)"
    )
    return "\n".join(lines)

"""Воспроизведение записанного маршрута без перебоев и лагов.

Как устроен вывод
-----------------
* **Частота вывода** (по умолчанию 250 Гц = каждые 4 мс) задаётся генератором
  ``DeadlineLoop``: момент кадра считается как ``t0 + k/rate`` — отставание
  никогда не накапливается, «догонялок» и рывков не возникает.
* Перед сном включается системный таймер 1 мс (``timeBeginPeriod``), последние
  микросекунды идут спин-ожиданием — иначе Windows может «проспать» 15 мс.
* Приоритет процесса поднимается (High / MMCSS «Games»), сборщик мусора на
  время прогона отключается, состояния подготовлены заранее — в горячем цикле
  нет аллокаций и файловых операций.
* Если задано сглаживание, оно применяется **на лету** в полярной форме
  (амплитуда + направление) — стики не «пилят», игра получает ровную дугу.
* Всё, что реально ушло в виртуальный геймпад, можно записать в телеметрию и
  затем посчитать по ней те же метрики плавности (``--telemetry``).
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .filters import StickSmoother
from .frame import PadState, mask_names
from .gamepad import VirtualPad, make_pad
from .metrics import analyze_runs, playback_verification
from .plan import Plan, PlanConfig, Sampler, build_plan, describe_plan
from .processing import ProcessConfig, process
from .session import Clip
from .timing import (
    DeadlineLoop,
    LoopStats,
    TimerResolution,
    enable_mmcss,
    freeze_gc,
    now,
    pin_cpu,
    set_process_priority,
)


@dataclass
class PlayConfig:
    """Настройки воспроизведения."""

    rate_hz: float = 250.0
    speed: float = 1.0
    loops: int = 0  # 0 = бесконечно
    loop_gap: float = 0.15
    deadzone: float = 0.0
    interp: str = "linear"
    game_fps: float = 60.0
    min_press_scale: float = 1.5
    # сглаживание
    filter_mode: str = "zero"  # off | zero | xy | polar
    smooth_hz: float = 15.0  # срез нуль-фазового сглаживания маршрута (без задержки)
    # сглаживание «на лету» (polar/xy) — с постоянным отставанием по времени
    min_cutoff: float = 1.6
    beta: float = 0.02
    beta_angle: float = 0.012
    # тайминги
    precise: bool = True
    lag_tolerance: float = 0.002
    entry_ramp_ms: float = 200.0
    exit_ramp_ms: float = 200.0
    lead_in: float = 0.0
    tail_hold_ms: float = 30.0
    countdown: float = 3.0
    # вывод
    pad_backend: str = "xinput"
    pad_deadzone: float = 0.0
    invert_y: bool = True
    # диагностика
    telemetry: bool = False
    telemetry_hz: float = 120.0
    priority: str = "high"
    mmcss: bool = True
    cpu: Optional[int] = None
    quiet: bool = False
    live: bool = True


@dataclass
class PlayResult:
    cycles: int = 0
    ticks: int = 0
    duration: float = 0.0
    loop_stats: Dict[str, Any] = field(default_factory=dict)
    work_us: Dict[str, float] = field(default_factory=dict)
    telemetry: Optional[Clip] = None
    planned: Optional[Clip] = None
    report: Dict[str, Any] = field(default_factory=dict)
    verification: Dict[str, Any] = field(default_factory=dict)
    aborted: bool = False
    dropped: int = 0
    markers: List[Tuple[float, str]] = field(default_factory=list)


class ConsoleKeys:
    """Неблокирующее чтение клавиш из консоли (работает и без окна игры).

    Клавиши: ``пробел``/``p`` — пауза, ``q``/``Esc`` — стоп, ``m`` — метка.
    """

    def __init__(self) -> None:
        self._enabled = True
        self._win = sys.platform.startswith("win")
        if self._win:
            try:
                import msvcrt  # noqa: F401  (проверка доступности)
            except Exception:
                self._enabled = False
        else:
            try:
                import select  # noqa: F401  (проверка доступности)
                import termios  # noqa: F401
                import tty  # noqa: F401

                if not sys.stdin.isatty():
                    self._enabled = False
            except Exception:
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def poll(self) -> Optional[str]:
        if not self._enabled:
            return None
        if self._win:
            import msvcrt

            chars: List[str] = []
            while msvcrt.kbhit():
                chars.append(msvcrt.getwch())
            return "".join(chars) if chars else None
        import select

        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        return sys.stdin.read(1) or None


class Player:
    """Проигрыватель маршрута: клип -> виртуальный геймпад."""

    def __init__(
        self,
        clip: Clip,
        config: Optional[PlayConfig] = None,
        *,
        process_cfg: Optional[ProcessConfig] = None,
    ) -> None:
        self.config = config or PlayConfig()
        self.raw_clip = clip
        self.process_cfg = process_cfg
        self.processed_stats: Dict[str, Any] = {}
        if process_cfg is not None:
            res = process(clip, process_cfg, source_name=clip.name)
            self.clip = res.clip
            self.processed_stats = res.stats
        else:
            self.clip = clip
        self.plan: Optional[Plan] = None
        self.pad: Optional[VirtualPad] = None

    # -- подготовка ---------------------------------------------------------
    def build(self) -> Plan:
        cfg = self.config
        plan_cfg = PlanConfig(
            rate_hz=cfg.rate_hz,
            speed=cfg.speed,
            loops=cfg.loops,
            loop_gap=cfg.loop_gap,
            deadzone=cfg.deadzone,
            interp=cfg.interp,
            game_fps=cfg.game_fps,
            min_press_scale=cfg.min_press_scale,
            lead_in_ms=cfg.lead_in * 1000.0,
            tail_hold_ms=cfg.tail_hold_ms,
            entry_ramp_ms=cfg.entry_ramp_ms,
            exit_ramp_ms=cfg.exit_ramp_ms,
        )
        self.plan = build_plan(self.clip, plan_cfg)
        if cfg.filter_mode == "zero":
            # маршрут известен целиком → сглаживаем без задержки
            self.plan.smooth_zero_phase(
                cfg.smooth_hz, deadzone=max(cfg.deadzone, 0.02)
            )
        return self.plan

    def _make_smoothers(self) -> Optional[Dict[str, StickSmoother]]:
        cfg = self.config
        if cfg.filter_mode in ("off", "none", "", "zero"):
            return None
        return {
            side: StickSmoother(
                spacing_hz=cfg.rate_hz,
                min_cutoff=cfg.min_cutoff,
                beta=cfg.beta,
                beta_angle=cfg.beta_angle,
                mode=cfg.filter_mode,
                deadzone=max(cfg.deadzone, 0.02),
            )
            for side in ("L", "R")
        }

    # -- воспроизведение ----------------------------------------------------
    def run(
        self,
        *,
        hotkeys: bool = True,
        keys: Any = None,
        on_event: Optional[Callable[[str], None]] = None,
    ) -> PlayResult:
        cfg = self.config
        plan = self.plan or self.build()
        if self.pad is None:
            self.pad = make_pad(cfg.pad_backend, deadzone=cfg.pad_deadzone, invert_y=cfg.invert_y)
        pad = self.pad

        result = PlayResult()
        stats = LoopStats("output")
        loop = DeadlineLoop(
            cfg.rate_hz,
            spin_margin=0.00025,
            precise=cfg.precise,
            stats=stats,
            lag_tolerance=cfg.lag_tolerance,
        )
        smoothers = self._make_smoothers()
        if keys is None and hotkeys:
            keys = ConsoleKeys()
        tele = Clip(meta={"name": "telemetry", "source": self.clip.name}) if cfg.telemetry else None
        tele_runs: List[int] = []  # начало каждого круга внутри телеметрии
        run_pending = True
        planned = Clip(meta={"name": "planned"}) if cfg.telemetry else None
        tele_every = max(1, int(round(cfg.rate_hz / max(1.0, cfg.telemetry_hz))))
        # участки плавного входа/выхода не входят в телеметрию: это намеренные
        # переходные процессы, а метрики должны описывать сам маршрут
        edge = int(round(max(cfg.entry_ramp_ms, cfg.exit_ramp_ms) / 1000.0 * cfg.rate_hz))
        edge += max(1, int(round(0.06 * cfg.rate_hz)))
        # для коротких маршрутов окно сужается пропорционально, но не исчезает
        edge = max(1, min(edge, max(1, plan.ticks // 5)))
        tele_from = edge
        tele_to = max(tele_from + 1, plan.ticks - edge)

        period = 1.0 / cfg.rate_hz
        dt = period
        paused = False
        aborted = False
        work: List[float] = []
        k = 0
        local = 0
        cycle = 0
        state = PadState()

        if not cfg.quiet:
            print(describe_plan(plan))
            print(
                f"Вывод: {pad.name} | приоритет процесса: {cfg.priority}"
                + (" | MMCSS «Games»" if cfg.mmcss else "")
                + " | точный таймер 1 мс"
            )
            if smoothers:
                print(
                    f"Сглаживание на лету: режим {cfg.filter_mode}, min-cutoff {cfg.min_cutoff} Гц, "
                    f"beta {cfg.beta}, beta(угол) {cfg.beta_angle}"
                )
            if cfg.telemetry:
                print(f"Телеметрия вывода: {cfg.telemetry_hz:.0f} Гц (для проверки плавности)")

        def emit(st: PadState) -> None:
            """Отправить состояние в геймпад."""
            pad.write(st)

        def emit_neutral() -> None:
            """Нейтраль + синхронизация фильтров (иначе после паузы будет рывок)."""
            if smoothers is not None:
                for sm in smoothers.values():
                    sm.reset(0.0, 0.0)
            pad.write(PadState())

        try:
            with TimerResolution(1):
                set_process_priority(cfg.priority)
                if cfg.mmcss:
                    enable_mmcss()
                if cfg.cpu is not None:
                    pin_cpu(cfg.cpu)
                pad.reset()
                if smoothers is not None:
                    for sm in smoothers.values():
                        sm.reset(0.0, 0.0)
                self._countdown(cfg.countdown if hotkeys or not cfg.quiet else 0.0)
                if not cfg.quiet:
                    print("▶ СТАРТ — переключитесь в игру и не трогайте мышь.")
                    print("   Пауза: пробел/p   Стоп: q/Esc   Метка: m")
                loop.start()
                with freeze_gc():
                    # «разгон»: не подавать движение, пока не включился точный таймер
                    for _ in range(plan.lead_ticks):
                        emit_neutral()
                        loop.wait_for(k)
                        k += 1
                    sampler = Sampler(plan)
                    while True:
                        t_work0 = now()
                        # --- управление --------------------------------------
                        if keys is not None:
                            ch = keys.poll()
                            if ch:
                                low = ch.lower()
                                if "\x1b" in ch or "q" in low:
                                    aborted = True
                                    if on_event:
                                        on_event("abort")
                                    break
                                if " " in ch or "p" in low:
                                    paused = not paused
                                    if on_event:
                                        on_event("pause" if paused else "resume")
                                if "m" in low:
                                    stamp = stats.frames * period
                                    result.markers.append((stamp, f"метка {stamp:.2f} с"))
                                    if on_event:
                                        on_event(f"marker {stamp:.2f}")
                        if paused:
                            emit_neutral()
                            time.sleep(0.02)
                            loop.t0 = now() - k * period  # пауза не «съедает» расписание
                            continue

                        # --- состояние ---------------------------------------
                        if local < plan.ticks:
                            state = sampler(local)  # что записано (план)
                            if smoothers is not None:
                                planned_state = state
                                lx, ly = smoothers["L"](state.lx, state.ly, dt)
                                rx, ry = smoothers["R"](state.rx, state.ry, dt)
                                state = PadState(lx, ly, rx, ry, state.lt, state.rt, state.btns)
                            else:
                                planned_state = state
                        else:
                            state = planned_state = PadState()

                        # --- вывод -------------------------------------------
                        emit(state)
                        t_work = now() - t_work0  # чистое время подготовки+отправки

                        # Телеметрия пишется только «внутри маршрута»: паузы между
                        # кругами и удержание нейтрали в неё не попадают, поэтому
                        # метрики плавности описывают сам маршрут, а не паузы.
                        if tele is not None and run_pending and local >= tele_from:
                            tele_runs.append(len(tele))
                            run_pending = False
                        if tele is not None and tele_from <= local < tele_to and (stats.frames % tele_every == 0):
                            stamp = stats.frames * period
                            tele.append(stamp, state)
                            if planned is not None:
                                planned.append(stamp, planned_state)

                        # --- расписание --------------------------------------
                        loop.wait_for(k)
                        work.append(t_work)
                        k += 1
                        local += 1
                        if local < plan.ticks:
                            if cfg.live and not cfg.quiet and k % max(1, int(cfg.rate_hz)) == 0:
                                self._live_line(stats, state, cycle, k, loop.dropped)
                            continue
                        # круг завершён
                        cycle += 1
                        if cfg.loops and cycle >= cfg.loops:
                            break
                        gap = plan.gap_ticks()
                        for _ in range(gap):
                            emit_neutral()
                            loop.wait_for(k)
                            k += 1
                        local = 0
                        run_pending = True
                        sampler.reset()
                    # финальное удержание нейтрали
                    for _ in range(plan.tail_ticks):
                        emit_neutral()
                        loop.wait_for(k)
                        k += 1
        except KeyboardInterrupt:
            aborted = True
        finally:
            pad.reset()
            pad.close()

        result.aborted = aborted
        result.cycles = cycle
        result.ticks = stats.frames
        result.dropped = loop.dropped
        result.duration = stats.frames * period
        result.loop_stats = stats.summary()
        result.work_us = {
            "mean": round(statistics.fmean(work) * 1e6, 1) if work else 0.0,
            "p95": round(_pct(work, 0.95) * 1e6, 1),
            "max": round(max(work) * 1e6, 1) if work else 0.0,
        }
        if tele is not None:
            tele.meta["duration"] = round(tele.duration, 4)
            if len(tele_runs) > 1:
                # границы кругов сохраняем в сам клип: иначе отчёт по файлу
                # телеметрии посчитал бы стык кругов как «рывок» стика
                tele.meta["runs"] = len(tele_runs)
                tele.meta["run_starts"] = list(tele_runs)
            result.telemetry = tele
            if len(tele) > 3:
                result.report = analyze_runs(tele, tele_runs)
            if planned is not None and len(planned) > 3:
                result.verification = playback_verification(planned, tele)
        return result

    # -- вспомогательное ----------------------------------------------------
    def _countdown(self, seconds: float) -> None:
        if seconds <= 0 or self.config.quiet:
            return
        print("Готовность: переключитесь в игру (окно игры должно быть активным)!")
        end = now() + seconds
        last = None
        while True:
            left = end - now()
            if left <= 0:
                break
            whole = int(math.ceil(left))
            if whole != last:
                print(f"  старт через {whole}…", end="\r", flush=True)
                last = whole
            time.sleep(0.02)
        print(" " * 32, end="\r")

    def _live_line(self, stats: LoopStats, state: PadState, cycle: int, k: int, dropped: int) -> None:
        s = stats.summary()
        cyc = f"{cycle + 1}/{self.config.loops}" if self.config.loops else f"{cycle + 1}"
        print(
            f"\r  t={k / self.config.rate_hz:7.2f}с круг {cyc:<8}"
            f"L=({state.lx:+.2f},{state.ly:+.2f}) R=({state.rx:+.2f},{state.ry:+.2f}) "
            f"LT={state.lt:.2f} RT={state.rt:.2f} [{'+'.join(mask_names(state.btns)) or '-':<8}] "
            f"джиттер {s['jitter_us']:5.0f} мкс, макс.опоздание {s['max_lag_us']:5.0f} мкс"
            + (f", пропусков {dropped}" if dropped else ""),
            end="",
            flush=True,
        )


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def format_play_result(result: PlayResult, *, title: str = "ИТОГИ ПРОГОНА") -> str:
    w = 80
    lines = ["", "=" * w, f" {title}", "=" * w]
    s = result.loop_stats
    lines.append(
        f" Кадров: {result.ticks} ({result.duration:.2f} с, кругов {result.cycles}) | "
        f"фактическая частота {s.get('actual_hz')} Гц"
    )
    lines.append(
        f" Стабильность вывода: джиттер {s.get('jitter_us')} мкс, максимальное опоздание "
        f"{s.get('max_lag_us')} мкс, пропущенных сроков {s.get('missed_deadlines')}"
        + (f", пропущено кадров {result.dropped}" if result.dropped else "")
    )
    wt = result.work_us
    lines.append(
        f" Работа на кадр: среднее {wt.get('mean')} мкс, p95 {wt.get('p95')} мкс, максимум {wt.get('max')} мкс"
    )
    if result.verification:
        v = result.verification
        if v.get("error"):
            lines.append(f" Сверка план/выход: {v['error']}")
        else:
            lines.append(
                f" Сверка план/выход: макс. отклонение по осям {max(v['axis_error_max'].values()):.4f}, "
                f"СКО {max(v['axis_error_rms'].values()):.4f} "
                f"(это работа сглаживания, не сбой), нажатий потеряно {v['buttons_missed']} "
                f"из {v['buttons_planned']}"
            )
    if result.markers:
        lines.append(" Метки: " + ", ".join(f"{t:.2f} с" for t, _ in result.markers))
    if result.aborted:
        lines.append(" Прогон прерван пользователем — все органы управления отпущены.")
    lines.append("=" * w)
    return "\n".join(lines)


__all__ = ["PlayConfig", "PlayResult", "Player", "ConsoleKeys", "format_play_result"]

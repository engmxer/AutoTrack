"""Точное время, планировщик кадров и приоритеты процесса.

Всё, что отвечает за «без перебоев и лагов»:

* монотонные часы на ``time.perf_counter`` (разрешение ~100 нс);
* ``timeBeginPeriod(1)`` на Windows — системный таймер 1 мс вместо 15.6 мс;
* «спин-ожидание» последних микросекунд перед дедлайном;
* повышение приоритета процесса/потока (Windows: REALTIME/HIGH + MMCSS «Games»);
* сбор статистики: пропущенные дедлайны, джиттер, максимальный разрыв.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

IS_WINDOWS = sys.platform.startswith("win")

now = time.perf_counter


# ---------------------------------------------------------------------------
# Разрешение системного таймера
# ---------------------------------------------------------------------------


class TimerResolution:
    """Контекстный менеджер ``timeBeginPeriod(1)`` (только Windows)."""

    def __init__(self, ms: int = 1) -> None:
        self.ms = max(1, int(ms))
        self._active = False

    def __enter__(self) -> "TimerResolution":
        if IS_WINDOWS:
            try:
                winmm = ctypes.WinDLL("winmm")  # type: ignore[attr-defined]
                if winmm.timeBeginPeriod(self.ms) == 0:
                    self._active = True
            except Exception:
                self._active = False
        return self

    def __exit__(self, *exc: object) -> None:
        if self._active and IS_WINDOWS:
            with contextlib.suppress(Exception):
                winmm = ctypes.WinDLL("winmm")  # type: ignore[attr-defined]
                winmm.timeEndPeriod(self.ms)
            self._active = False

    @property
    def active(self) -> bool:
        return self._active


# ---------------------------------------------------------------------------
# Приоритеты
# ---------------------------------------------------------------------------

PRIORITY_CLASSES = {
    "normal": 0x00000020,
    "above_normal": 0x00008000,
    "high": 0x00000080,
    "realtime": 0x00000100,
}

THREAD_PRIORITIES = {
    "normal": 0,
    "above_normal": 1,
    "highest": 2,
    "time_critical": 15,
}


def set_process_priority(level: str = "high") -> bool:
    """Повысить приоритет процесса. Возвращает ``True`` при успехе."""
    level = level.lower()
    try:
        if IS_WINDOWS:
            value = PRIORITY_CLASSES.get(level)
            if value is None:
                return False
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), value))
        nice = {"normal": 0, "above_normal": -5, "high": -10, "realtime": -15}.get(level)
        if nice is None:
            return False
        os.nice(nice)
        return True
    except Exception:
        return False


def set_thread_priority(level: str = "time_critical") -> bool:
    """Повысить приоритет текущего потока."""
    level = level.lower()
    try:
        if IS_WINDOWS:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            handle = kernel32.GetCurrentThread()
            return bool(kernel32.SetThreadPriority(handle, THREAD_PRIORITIES.get(level, 15)))
        nice = {"normal": 0, "above_normal": -5, "high": -10, "time_critical": -15}.get(level)
        if nice is None:
            return False
        os.nice(nice)
        return True
    except Exception:
        return False


def enable_mmcss(task: str = "Games") -> bool:
    """MMCSS-регистрация потока (Windows) — защита от «придушивания» планировщиком."""
    if not IS_WINDOWS:
        return False
    try:
        avrt = ctypes.WinDLL("avrt")  # type: ignore[attr-defined]
        task_index = ctypes.c_ulong()
        handle = avrt.AvSetMmThreadCharacteristicsW(ctypes.c_wchar_p(task), ctypes.byref(task_index))
        return bool(handle)
    except Exception:
        return False


def pin_cpu(index: Optional[int]) -> bool:
    """Привязать текущий поток к ядру (если поддерживается)."""
    if index is None or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, {index})  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


@contextlib.contextmanager
def freeze_gc():
    """Отключает сборщик мусора на время горячего цикла (борьба с микро-стопами)."""
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


# ---------------------------------------------------------------------------
# Ожидание дедлайна
# ---------------------------------------------------------------------------


def sleep_until(deadline: float, spin_margin: float = 0.0003, coarse: float = 0.0012) -> float:
    """Ждать момента ``deadline`` (``time.perf_counter``).

    Сначала ``sleep``, затем короткое «спин»-ожидание, чтобы не «проспать»
    дедлайн на 1–15 мс. Возвращает момент фактического выхода (для статистики).
    """
    remaining = deadline - now()
    if remaining > coarse:
        time.sleep(remaining - coarse)
        remaining = deadline - now()
    # Активное ожидание последних микросекунд
    while remaining > 0.0:
        remaining = deadline - now()
        if remaining > spin_margin:
            time.sleep(0)
    return now()


@dataclass
class LoopStats:
    """Статистика цикла вывода: ровно то, что нужно, чтобы доказать «нет лагов»."""

    label: str = "loop"
    frames: int = 0
    warmup: int = 5  # первые кадры не учитываем (разгон таймера, ленивые импорты)
    total_wait_us: float = 0.0
    periods: List[float] = field(default_factory=list)
    lags: List[float] = field(default_factory=list)
    _last: Optional[float] = None

    def tick(self, lag: float) -> None:
        """``lag`` — на сколько мы опоздали относительно планового момента (<0 = раньше)."""
        self.frames += 1
        if self.frames <= self.warmup:
            self._last = now()
            return
        self.lags.append(lag)
        t = now()
        if self._last is not None:
            self.periods.append(t - self._last)
        self._last = t

    # -- сводка ------------------------------------------------------------
    @property
    def jitter_us(self) -> float:
        if len(self.periods) < 2:
            return 0.0
        return statistics.pstdev(self.periods) * 1e6

    @property
    def max_lag_us(self) -> float:
        return max((x for x in self.lags), default=0.0) * 1e6

    @property
    def missed(self) -> int:
        """Число кадров, опоздавших больше чем на половину периода (грубый сбой ритма)."""
        if len(self.periods) < 2:
            return 0
        target = statistics.median(self.periods)
        return sum(1 for lag in self.lags if lag > target * 0.5 + 0.002)

    def summary(self) -> dict:
        period = statistics.median(self.periods) if self.periods else 0.0
        return {
            "label": self.label,
            "frames": self.frames,
            "actual_hz": round(1.0 / period, 2) if period else 0.0,
            "jitter_us": round(self.jitter_us, 1),
            "max_lag_us": round(self.max_lag_us, 1),
            "missed_deadlines": self.missed,
            "duration_s": round(sum(self.periods) + (period or 0.0), 3) if self.periods else 0.0,
        }

    def report(self) -> str:
        s = self.summary()
        return (
            f"[{s['label']}] кадров={s['frames']} фактическая частота={s['actual_hz']} Гц "
            f"джиттер={s['jitter_us']} мкс макс.опоздание={s['max_lag_us']} мкс "
            f"сбоев={s['missed_deadlines']}"
        )


class DeadlineLoop:
    """Генератор плановых моментов времени для фиксированной частоты.

    Использование::

        loop = DeadlineLoop(250.0, spin_margin=3e-4)
        for k in loop.count(300):
            ...  # отправить кадр k
            loop.wait_for(k)
    """

    def __init__(
        self,
        rate_hz: float,
        *,
        spin_margin: float = 0.0003,
        precise: bool = True,
        stats: Optional[LoopStats] = None,
        lag_tolerance: float = 0.0,
    ) -> None:
        self.rate = float(rate_hz)
        self.period = 1.0 / self.rate
        self.spin_margin = spin_margin
        self.precise = precise
        self.stats = stats if stats is not None else LoopStats()
        self.overshoot_ema = 0.0005
        self.lag_tolerance = lag_tolerance
        self.dropped = 0

    def start(self) -> float:
        self.t0 = now()
        return self.t0

    def deadline(self, k: int) -> float:
        return self.t0 + k * self.period

    def wait_for(self, k: int) -> float:
        """Дождаться планового момента кадра ``k``. Возвращает фактическое время."""
        deadline = self.deadline(k)
        if self.precise:
            margin = min(max(self.overshoot_ema, 0.0001), 0.003)
            t = sleep_until(deadline, spin_margin=margin)
        else:
            remaining = deadline - now()
            if remaining > 0:
                time.sleep(remaining)
            t = now()
        lag = t - deadline
        self.overshoot_ema = 0.9 * self.overshoot_ema + 0.1 * max(lag, 0.0)
        self.stats.tick(lag)
        return t

    def skip_late(self, k: int) -> int:
        """Пропустить кадры, которые уже «просрочены» (защита от накопления долга).

        Возвращает индекс следующего кадра. Если система не успевает, лучше
        пропустить кадр ввода, чем «догонять» и получить рывок в игре.
        """
        if self.lag_tolerance <= 0:
            return k
        while (now() - self.deadline(k)) > self.lag_tolerance:
            k += 1
            self.dropped += 1
        return k

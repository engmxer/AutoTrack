"""Фильтры и обработка сигнала."""

import math

import pytest

from autotrack.dsp import highpass, lowpass, median_filter, moving_average, resample, uniform_grid
import random

from autotrack.filters import OneEuroFilter, StickSmoother, unwrap_angle, zero_phase_stick


def sine(n=1000, f=1.0, fs=250.0, noise=0.0, seed=0):
    import random

    random.seed(seed)
    return [math.sin(2 * math.pi * f * i / fs) + random.gauss(0, noise) for i in range(n)]


def test_moving_average_keeps_dc():
    x = [1.0] * 50
    assert moving_average(x, 5) == pytest.approx([1.0] * 50)


def test_median_filter_removes_spike():
    x = [0.0] * 21
    x[10] = 5.0
    y = median_filter(x, 3)
    assert max(y) == 0.0


def test_lowpass_attenuates_high_frequency():
    fast = sine(n=1000, f=60.0)
    slow = sine(n=1000, f=2.0)
    fast_f = lowpass(fast, 15.0, 250.0)
    assert max(abs(v) for v in fast_f) < 0.1
    assert max(abs(v) for v in lowpass(slow, 15.0, 250.0)) > 0.9


def test_highpass_keeps_noise_and_drops_slow():
    slow = sine(n=1000, f=1.0)
    noise = [((-1) ** i) * 0.1 for i in range(1000)]
    hp_slow = highpass(slow, 5.0, 250.0)
    hp_noise = highpass(noise, 5.0, 250.0)
    # в середине медленный сигнал подавлен полностью, на краях остаётся
    # небольшой переходный процесс (свойство любого filtfilt)
    assert max(abs(v) for v in hp_slow[100:-100]) < 0.01
    assert max(abs(v) for v in hp_slow) < 0.1
    assert max(abs(v) for v in hp_noise) > 0.07


def test_resample_linear_midpoint():
    src_t = [0.0, 1.0]
    vals = [0.0, 10.0]
    out = resample(vals, src_t, [0.5], interp="linear")
    assert out[0] == pytest.approx(5.0)


def test_uniform_grid():
    grid = uniform_grid(1.0, 100.0)
    assert len(grid) == 101
    assert grid[1] == pytest.approx(0.01)


def test_one_euro_removes_noise():
    """Фильтр должен уменьшать дрожание (а не «сдвигать» сигнал: у фильтра есть лаг)."""
    from autotrack.metrics import noise_level

    fs = 250.0
    noisy = sine(n=1500, f=1.0, fs=fs, noise=0.05, seed=1)
    f = OneEuroFilter(min_cutoff=1.5, beta=0.01)
    out = [f(v, 1.0 / fs) for v in noisy]
    before = noise_level(noisy, fs)["noise_rms"]
    after = noise_level(out, fs)["noise_rms"]
    assert after < before * 0.5, (before, after)
    # медленная составляющая при этом сохраняется
    assert max(abs(v) for v in out[200:]) > 0.8


def test_stick_smoother_keeps_direction():
    sm = StickSmoother(spacing_hz=250.0, min_cutoff=2.0, beta=0.02, mode="polar")
    x, y = 0.0, 0.0
    for i in range(500):
        t = i / 250.0
        x, y = sm(0.7 * math.cos(t * 2), 0.7 * math.sin(t * 2), 1 / 250.0)
    assert math.hypot(x, y) == pytest.approx(0.7, abs=0.05)


def test_stick_smoother_reset():
    sm = StickSmoother(spacing_hz=250.0, mode="polar")
    for i in range(50):
        sm(0.5, 0.5, 0.004)
    sm.reset(0.0, 0.0)
    x, y = sm(0.0, 0.0, 0.004)
    assert abs(x) < 1e-9 and abs(y) < 1e-9


def test_unwrap_angle():
    vals = [math.radians(a) for a in (170, 175, -180, -175, -170)]
    un = unwrap_angle(vals)
    assert all(abs(b - a) < 0.2 for a, b in zip(un, un[1:]))


# --- нуль-фазовое сглаживание маршрута (режим «zero») ---------------------


def test_zero_phase_stick_has_no_delay():
    """Задержки нет: фронт ступеньки остаётся на своём месте."""
    fs = 250.0
    n = 400
    lx = [0.0] * 200 + [0.8] * (n - 200)
    ox, _ = zero_phase_stick(lx, [0.0] * n, fs, 15.0)
    # середина фронта ровно там, где ступенька (первый проход набегает,
    # второй — отстаёт, поэтому сдвига не остаётся)
    cross = next(i for i, v in enumerate(ox) if v >= 0.4)
    assert abs(cross - 200) <= 1, cross
    assert abs(ox[-1] - 0.8) < 0.03
    # переход короткий: за 6 сэмплов (24 мс) сигнал проходит почти всю ступеньку
    assert ox[200 - 3] < 0.35 and ox[200 + 3] > 0.45


def test_zero_phase_keeps_short_flicks():
    """Быстрый флик (наклон «горкой» 16/32/16 мс) проходит не искажаясь."""
    fs = 250.0
    n = 400
    lx = []
    for i in range(n):
        t = (i - 150) / fs * 1000.0  # мс от начала флика
        if 0 <= t < 16:
            v = 0.5 - 0.5 * math.cos(math.pi * t / 16.0)
        elif 16 <= t < 48:
            v = 1.0
        elif 48 <= t < 64:
            v = 0.5 + 0.5 * math.cos(math.pi * (t - 48) / 16.0)
        else:
            v = 0.0
        lx.append(v)
    ox, _ = zero_phase_stick(lx, [0.0] * n, fs, 15.0)
    assert max(ox) > 0.95, max(ox)


def test_zero_phase_suppresses_single_sample_spikes():
    """Одиночный выброс (4 сэмпла, 16 мс) — это шум, он гасится."""
    fs = 250.0
    n = 400
    lx = [0.0] * 150 + [1.0] * 4 + [0.0] * (n - 154)
    ox, _ = zero_phase_stick(lx, [0.0] * n, fs, 15.0)
    assert max(ox) < 0.6, max(ox)


def test_zero_phase_reduces_jitter():
    """Дрожание вокруг удержания уменьшается, центр не смещается."""
    fs = 250.0
    rnd = random.Random(3)
    lx = [0.5 + rnd.uniform(-0.01, 0.01) for _ in range(600)]
    ox, _ = zero_phase_stick(lx, [0.0] * 600, fs, 15.0)
    assert max(abs(v - 0.5) for v in ox) < 0.006
    assert abs(sum(ox) / len(ox) - 0.5) < 0.005

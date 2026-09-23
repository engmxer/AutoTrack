"""Пост-обработка записи."""

import math
import random

import pytest

from autotrack.frame import PadState
from autotrack.processing import ProcessConfig, describe_process, process
from autotrack.session import Clip


def noisy_clip(n=1000, rate=250.0, noise=0.02, drop_at=None):
    random.seed(11)
    clip = Clip(meta={"name": "noisy"})
    t = 0.0
    for i in range(n):
        t = i / rate
        clip.append(
            t,
            PadState(
                lx=0.6 * math.cos(t * 2) + random.gauss(0, noise),
                ly=0.6 * math.sin(t * 2) + random.gauss(0, noise),
                rt=1.0 if 1.0 < t < 1.2 else 0.0,
                btns=4 if 2.0 < t < 2.05 else 0,
            ),
        )
    if drop_at:
        # имитируем пропуск ввода: выкидываем пачку сэмплов
        idx = int(drop_at * rate)
        keep = [(clip.t[i], clip.frame(i)) for i in range(len(clip)) if not (idx <= i < idx + 40)]
        clip = Clip(meta=dict(clip.meta))
        for t, st in keep:
            clip.append_raw(t, st)
    return clip


def test_process_smooths_noise():
    clip = noisy_clip()
    res = process(clip, ProcessConfig(lowpass_hz=14.0, resample_hz=0.0), source_name="t")
    from autotrack.metrics import analyze

    before = analyze(clip)["sticks"]["L"]["noise_pct"]
    after = analyze(res.clip)["sticks"]["L"]["noise_pct"]
    assert after < before * 0.6, (before, after)


def test_process_resamples_to_uniform_grid():
    clip = noisy_clip()
    res = process(clip, ProcessConfig(resample_hz=120.0))
    assert res.stats["output_samples"] != len(clip)
    assert res.clip.rate_hz == pytest.approx(120.0, rel=0.02)


def test_process_keeps_duration():
    clip = noisy_clip()
    res = process(clip, ProcessConfig(resample_hz=120.0))
    assert res.clip.duration == pytest.approx(clip.duration, abs=0.05)


def test_process_keeps_buttons():
    clip = noisy_clip()
    res = process(clip, ProcessConfig(resample_hz=120.0))
    assert res.clip.button_runs(["X"]), "короткое нажатие потерялось"


def test_trigger_ramp_expands_digital_trigger():
    clip = noisy_clip()
    res = process(clip, ProcessConfig(lowpass_hz=0.0, resample_hz=0.0, trigger_expand=True, trigger_ramp_ms=40.0))
    assert min(res.clip.rt) == 0.0
    mids = [v for v in res.clip.rt if 0.05 < v < 0.95]
    assert len(mids) >= 4, "курок не разглажен"


def test_process_detects_gaps():
    clip = noisy_clip(drop_at=1.5)
    res = process(clip, ProcessConfig(resample_hz=0.0))
    assert res.stats["gaps"] >= 1
    assert res.stats["gap_max_s"] > 0.01


def test_trim_idle_removes_silence():
    clip = Clip(meta={"name": "idle"})
    for i in range(1000):
        t = i / 250.0
        active = 1.5 < t < 2.5
        clip.append(t, PadState(lx=0.5 if active else 0.0, ly=0.0))
    res = process(clip, ProcessConfig(trim_idle=0.1, resample_hz=0.0))
    assert res.clip.duration < clip.duration
    assert res.clip.duration > 0.9


def test_describe_process():
    clip = noisy_clip(400)
    res = process(clip)
    text = describe_process(res.stats)
    assert "Обработка" in text


def test_deadzone_option():
    clip = Clip(meta={"name": "d"})
    for i in range(200):
        clip.append(i / 250.0, PadState(lx=0.03, ly=0.03, rx=0.9))
    res = process(clip, ProcessConfig(deadzone=0.05, lowpass_hz=0.0, resample_hz=0.0))
    assert res.clip.lx[10] == 0.0
    assert res.clip.rx[10] > 0.8


def test_grid_is_regular_with_runs():
    """Сетка оценивается внутри кругов: пауза между ними — не разброс шага."""
    from autotrack.processing import grid_is_regular

    # два «круга» по 0.1 с с шагом 4 мс и паузой 0.5 с между ними
    t = [i * 0.004 for i in range(25)] + [0.6 + i * 0.004 for i in range(25)]
    assert not grid_is_regular(t)
    assert grid_is_regular(t, runs=[0, 25])


def test_process_keeps_uniform_grid():
    """Если сетка уже ровная, ресемпл на близкую частоту не делается."""
    from autotrack.processing import ProcessConfig, process

    rate = 125.0
    clip = Clip(meta={"name": "even"})
    for i in range(400):
        clip.append(i / rate, PadState(lx=0.4, ly=0.1))
    res = process(clip, ProcessConfig(median=0, lowpass_hz=0.0, resample_hz=120.0))
    assert res.stats.get("resampled_to") is None
    assert len(res.clip) == len(clip)

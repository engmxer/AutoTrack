"""План вывода: тики, интерполяция, минимальная длительность нажатий."""

import math

import pytest

from autotrack.frame import PadState, mask
from autotrack.plan import PlanConfig, Sampler, build_plan, describe_plan
from autotrack.session import Clip


def clip_with_press(press=0.004, rate=250.0, n=500):
    """Клип с очень коротким нажатием (1 сэмпл)."""
    clip = Clip(meta={"name": "press"})
    for i in range(n):
        t = i / rate
        btns = mask(["A"]) if abs(t - 1.0) < press / 2 else 0
        clip.append(t, PadState(lx=0.3 * math.cos(t * 4), ly=0.3 * math.sin(t * 4), btns=btns))
    return clip


def test_plan_ticks_and_rate():
    clip = clip_with_press()
    plan = build_plan(clip, PlanConfig(rate_hz=250.0, speed=1.0))
    assert plan.ticks == pytest.approx(499, abs=2)
    assert plan.duration == pytest.approx(clip.duration, abs=0.02)


def test_plan_speed_scales_duration():
    clip = clip_with_press()
    plan = build_plan(clip, PlanConfig(speed=2.0))
    assert plan.duration == pytest.approx(clip.duration / 2, abs=0.05)


def test_min_press_is_stretched():
    clip = clip_with_press(press=0.004)  # один сэмпл = 1 тик
    plan = build_plan(clip, PlanConfig(rate_hz=250.0, game_fps=60.0, min_press_scale=1.5))
    down = sum(1 for st in plan.states if st.btns & mask(["A"]))
    assert down >= 5, down
    # и кнопка не растянулась до «половины записи»
    assert down < 30


def test_trigger_pulse_stretched():
    clip = Clip(meta={"name": "rt"})
    for i in range(500):
        t = i / 250.0
        clip.append(t, PadState(rt=1.0 if abs(t - 1.0) < 0.002 else 0.0))
    plan = build_plan(clip)
    active = sum(1 for st in plan.states if st.rt > 0.04)
    assert active >= 5


def test_deadzone_zeroes_center():
    clip = Clip(meta={"name": "dz"})
    for i in range(300):
        clip.append(i / 250.0, PadState(lx=0.01, ly=-0.02, rx=0.5, ry=0.5))
    plan = build_plan(clip, PlanConfig(deadzone=0.05, stick_deadzone_scale=False))
    assert plan.states[10].lx == 0.0
    assert plan.states[10].rx == pytest.approx(0.5)


def test_sampler_interpolates_between_points():
    clip = Clip(meta={"name": "lin"})
    clip.append(0.0, PadState(lx=0.0))
    clip.append(1.0, PadState(lx=1.0))
    plan = build_plan(clip, PlanConfig(rate_hz=100.0))
    s = Sampler(plan)
    mid = s(plan.ticks // 2)
    assert 0.1 < mid.lx < 0.9
    assert s(plan.ticks - 1).lx > mid.lx


def test_cubic_interpolation_clamped():
    clip = Clip(meta={"name": "cub"})
    for i in range(50):
        clip.append(i / 50.0, PadState(lx=1.0 if i % 2 else -1.0))
    plan = build_plan(clip, PlanConfig(rate_hz=200.0, interp="cubic"))
    assert all(-1.0 <= st.lx <= 1.0 for st in plan.states)


def test_iter_states_loops():
    clip = clip_with_press(n=100)
    cfg = PlanConfig(rate_hz=100.0, loops=3, loop_gap=0.1)
    plan = build_plan(clip, cfg)
    got = list(plan.iter_states())
    cycles = {c for _k, c, _st in got}
    assert cycles == {0, 1, 2}
    assert plan.total_ticks() == pytest.approx(len(got), abs=2)


def test_describe_plan():
    plan = build_plan(clip_with_press(n=100))
    text = describe_plan(plan)
    assert "План" in text and "Гц" in text


def test_plan_to_clip_roundtrip():
    plan = build_plan(clip_with_press(n=100), PlanConfig(rate_hz=250.0))
    clip = plan.to_clip(loops=1)
    assert len(clip) > 90
    assert clip.duration == pytest.approx(plan.duration, abs=0.05)

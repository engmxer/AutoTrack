"""Воспроизведение: точность тайминга, вывод, телеметрия, проверка."""

import math
import os
import time

import pytest

from autotrack.frame import PadState
from autotrack.gamepad import DryRunPad, NullPad, make_pad, xinput_mask
from autotrack.player import PlayConfig, Player, format_play_result
from autotrack.processing import ProcessConfig
from autotrack.session import Clip
from autotrack.timing import DeadlineLoop, TimerResolution, sleep_until


def make_clip(duration=1.0, rate=250.0, name="play"):
    clip = Clip(meta={"name": name})
    n = int(duration * rate)
    for i in range(n):
        t = i / rate
        clip.append(t, PadState(lx=0.5 * math.cos(t * 2), ly=0.5 * math.sin(t * 2), btns=1 if 0.5 < t < 0.6 else 0))
    return clip


def test_sleep_until_precision():
    target = time.perf_counter() + 0.02
    t = sleep_until(target)
    assert abs(t - target) < 0.004


def test_deadline_loop_stats():
    loop = DeadlineLoop(200.0, spin_margin=0.00025)
    with TimerResolution(1):
        loop.start()
        for k in range(200):
            loop.wait_for(k)
    s = loop.stats.summary()
    assert s["frames"] == 200
    assert s["actual_hz"] == pytest.approx(200.0, rel=0.05)
    assert s["jitter_us"] < 3000.0


def test_make_pad_kinds():
    assert isinstance(make_pad("dry"), DryRunPad)
    assert isinstance(make_pad("null"), NullPad)
    with pytest.raises(ValueError):
        make_pad("нет-такого")


def test_xinput_mask_maps_buttons():
    assert xinput_mask(1) == 0x1000  # A
    assert xinput_mask(1 << 5) == 0x0200  # RB


def test_player_dry_run_output():
    cfg = PlayConfig(rate_hz=250.0, loops=1, countdown=0, quiet=True, pad_backend="dry", live=False, telemetry=True)
    player = Player(make_clip(0.5), cfg, process_cfg=ProcessConfig(lowpass_hz=20.0, resample_hz=250.0))
    plan = player.build()
    assert plan.ticks > 100
    res = player.run(hotkeys=False)
    assert res.ticks >= plan.ticks
    assert res.loop_stats["missed_deadlines"] <= 5
    assert res.work_us["p95"] < 2000.0  # микросекунды на кадр
    assert res.report and res.report.get("samples", 0) > 10
    assert res.verification["buttons_missed"] == 0
    assert len(player.pad.rows) > 50


def test_player_loops_and_neutral_between():
    cfg = PlayConfig(rate_hz=200.0, loops=2, loop_gap=0.1, countdown=0, quiet=True, pad_backend="dry", live=False)
    res = Player(make_clip(0.4), cfg).run(hotkeys=False)
    assert res.cycles == 2


class FakeKeys:
    """Имитация консоли: N пустых опросов, затем нажатие."""

    enabled = True

    def __init__(self, key="q", after=20):
        self.key = key
        self.after = after
        self.calls = 0

    def poll(self):
        self.calls += 1
        if self.calls > self.after:
            return self.key
        return None


def test_player_stops_on_hotkey():
    cfg = PlayConfig(rate_hz=200.0, loops=0, countdown=0, quiet=True, pad_backend="dry", live=False)
    player = Player(make_clip(0.5), cfg)
    keys = FakeKeys("q", after=25)
    res = player.run(keys=keys)
    assert res.aborted
    assert 20 <= res.ticks <= 60, res.ticks


def test_player_pause_hotkey():
    cfg = PlayConfig(rate_hz=200.0, loops=1, countdown=0, quiet=True, pad_backend="dry", live=False)

    class PauseThenStop(FakeKeys):
        def poll(self):
            self.calls += 1
            if self.calls == 10:
                return " "
            if self.calls == 12:
                return " "
            if self.calls > 40:
                return "q"
            return None

    res = Player(make_clip(0.4), cfg).run(keys=PauseThenStop())
    assert res.ticks > 0


def test_polar_filter_reproduces_clean_rotation():
    """Полярное сглаживание не должно портить ровное вращение стика."""
    clip = Clip(meta={"name": "circle"})
    for i in range(1000):  # 4 с, 1 оборот в секунду
        t = i / 250.0
        clip.append(t, PadState(lx=0.6 * math.cos(2 * math.pi * t), ly=0.6 * math.sin(2 * math.pi * t)))
    for mode in ("polar", "off"):
        cfg = PlayConfig(
            rate_hz=250.0, loops=1, countdown=0, quiet=True, pad_backend="dry", live=False,
            telemetry=True, telemetry_hz=250.0, filter_mode=mode,
        )
        res = Player(clip, cfg).run(hotkeys=False)
        score = res.report["smoothness_score"]
        noise = res.report["sticks"]["L"]["noise_pct"]
        assert score > 85, (mode, score, noise)
        assert noise < 0.2, (mode, noise)


def test_entry_exit_ramps_are_smooth():
    """Плавный вход/выход: в конце круга стик возвращается к нулю без разрыва."""
    clip = Clip(meta={"name": "square"})
    for i in range(500):
        clip.append(i / 250.0, PadState(lx=0.9, ly=0.0))
    cfg = PlayConfig(rate_hz=250.0, loops=1, countdown=0, quiet=True, pad_backend="dry", live=False)
    plan = Player(clip, cfg).build()
    assert abs(plan.states[0].lx) < 0.05, plan.states[0].lx
    assert abs(plan.states[-1].lx) < 0.05, plan.states[-1].lx
    assert abs(plan.states[len(plan.states) // 2].lx) > 0.85


def test_format_play_result():
    cfg = PlayConfig(rate_hz=250.0, loops=1, countdown=0, quiet=True, pad_backend="dry", live=False, telemetry=True)
    res = Player(make_clip(0.3), cfg).run(hotkeys=False)
    text = format_play_result(res)
    assert "ИТОГИ" in text and "джиттер" in text


def test_telemetry_saved_and_matched():
    cfg = PlayConfig(
        rate_hz=250.0, loops=1, countdown=0, quiet=True, pad_backend="dry", live=False,
        telemetry=True, telemetry_hz=250.0,
    )
    res = Player(make_clip(0.4), cfg).run(hotkeys=False)
    assert res.telemetry is not None and len(res.telemetry) > 50
    assert res.verification["axis_error_max"]["lx"] < 0.6


def test_filter_off_matches_plan():
    cfg = PlayConfig(
        rate_hz=250.0, loops=1, countdown=0, quiet=True, pad_backend="dry", live=False,
        telemetry=True, telemetry_hz=250.0, filter_mode="off",
    )
    res = Player(make_clip(0.3), cfg).run(hotkeys=False)
    # без сглаживания вывод должен почти совпадать с планом
    assert res.verification["axis_error_max"]["lx"] < 0.05


def test_telemetry_report_matches_saved_clip():
    """Отчёт по сохранённой телеметрии не должен расходиться с отчётом прогона."""
    import tempfile

    from autotrack.cli import cmd_report, build_parser
    from autotrack.metrics import analyze
    from autotrack.session import Clip

    clip = make_clip(1.0)
    cfg = PlayConfig(
        rate_hz=250.0, loops=2, countdown=0, quiet=True, pad_backend="dry", live=False,
        telemetry=True, telemetry_hz=250.0,
    )
    res = Player(clip, cfg).run(hotkeys=False)
    run_score = res.report["smoothness_score"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tele.atk.json")
        res.telemetry.save(path)
        loaded = Clip.load(path)
        assert run_score - analyze(loaded)["smoothness_score"] < 1.0
        args = build_parser().parse_args(["report", path])
        assert cmd_report(args) == 0

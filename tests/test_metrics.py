"""Метрики плавности: чувствительность и сводная оценка."""

import math
import random

from autotrack.frame import PadState
from autotrack.metrics import (
    analyze,
    analyze_runs,
    count_sustained_flips,
    format_report,
    moving_average,
    noise_level,
    playback_verification,
    stick_metrics,
    unwrap_angles,
)
from autotrack.session import Clip


def circle_clip(noise=0.0, n=2000, rate=250.0, seed=7, name=None):
    random.seed(seed)
    clip = Clip(meta={"name": name or f"circle noise={noise}"})
    for i in range(n):
        t = i / rate
        ph = t * 2.0
        clip.append(
            t,
            PadState(
                lx=0.75 * math.cos(ph) + random.gauss(0, noise),
                ly=0.75 * math.sin(ph) + random.gauss(0, noise),
            ),
        )
    return clip


def test_clean_circle_is_smooth():
    report = analyze(circle_clip(0.0))
    assert report["smoothness_score"] > 85
    assert report["sticks"]["L"]["noise_pct"] < 0.2


def test_noise_lowers_score():
    scores = [analyze(circle_clip(nz))["smoothness_score"] for nz in (0.0, 0.002, 0.01, 0.03)]
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] - scores[-1] > 30


def test_noise_metric_scales_linearly():
    levels = []
    for nz in (0.002, 0.01, 0.03):
        st = analyze(circle_clip(nz))["sticks"]["L"]
        levels.append(st["noise_pct"])
    assert levels[0] < levels[1] < levels[2]
    assert levels[2] < 10.0  # в процентах от полной шкалы


def test_stick_metrics_fields():
    st = stick_metrics(
        *[circle_clip(0.0).axis("lx"), circle_clip(0.0).axis("ly"), circle_clip(0.0).t], "L"
    )
    for key in (
        "speed_mean", "speed_cv", "speed_jitter", "noise_pct", "angular_speed_mean",
        "angular_accel_p95", "direction_jitter_deg", "direction_flips_per_s",
        "smoothness_score", "returns_to_center",
    ):
        assert key in st


def test_flat_stick_scores_perfect():
    clip = Clip()
    for i in range(500):
        clip.append(i / 250.0, PadState())
    report = analyze(clip)
    assert report["smoothness_score"] > 95


def test_sustained_flips_detects_zigzag():
    omega = [1.0] * 10 + [-1.0] * 10 + [1.0] * 10
    assert count_sustained_flips(omega) == 2
    tiny = [0.1] * 20 + [-0.1] * 20  # слабее порога — не считаем
    assert count_sustained_flips(tiny) == 0


def test_unwrap_and_noise_level_helpers():
    vals = [math.radians(a) for a in (179, -179, -178)]
    un = unwrap_angles(vals)
    assert un[1] > un[0]
    x = [math.sin(i / 50.0) for i in range(1000)]
    lo = noise_level(x, 250.0)
    assert lo["noise_rms"] < 0.01


def test_moving_average_zero_phase():
    x = [math.sin(2 * math.pi * i / 250.0) for i in range(500)]
    y = moving_average(x, 9)
    shift = max(range(-5, 6), key=lambda k: sum(y[i] * x[i + k] for i in range(5, 495)))
    assert abs(shift) <= 1


def test_report_text():
    text = format_report(analyze(circle_clip(0.0)))
    assert "ПЛАВНОСТЬ" in text
    assert "стик" in text.lower()


def test_playback_verification():
    planned = circle_clip(0.0, name="p")
    emitted = circle_clip(0.0, name="e")
    v = playback_verification(planned, emitted)
    assert v["axis_error_max"]["lx"] < 1e-3
    assert v["buttons_planned"] == 0


def test_short_press_detected():
    clip = Clip()
    rate = 250.0
    for i in range(500):
        t = i / rate
        clip.append(t, PadState(btns=1 if 0.4 < t < 0.41 else 0))
    report = analyze(clip)
    assert len(report["short_presses"]) == 1


def test_analyze_runs_ignores_loop_seam():
    """Разрыв на стыке кругов — не дрожание стика: круги считаются отдельно."""
    fs = 250.0
    per_loop = 500
    clip = Clip(meta={"name": "two_loops"})
    for i in range(2 * per_loop):
        ph = 2 * math.pi * (i % per_loop) / per_loop + (math.pi if i >= per_loop else 0.0)
        clip.append(i / fs, PadState(lx=0.6 * math.cos(ph), ly=0.6 * math.sin(ph)))
    whole = analyze(clip)["sticks"]["L"]["noise_pct"]
    split = analyze_runs(clip, [0, per_loop])
    assert split["runs"] == 2
    noise = split["sticks"]["L"]["noise_pct"]
    assert noise < whole / 5, (noise, whole)
    assert split["sticks"]["L"]["smoothness_score"] > 85

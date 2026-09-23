"""Кадр состояния и формат клипа."""

import json
import math
import os

import pytest

from autotrack.frame import PadState, bit, mask, mask_names
from autotrack.session import Clip, concat


def test_button_bits():
    assert mask(["A", "B"]) == (1 << 0) | (1 << 1)
    assert bit("a") == 1
    names = mask_names(mask(["A", "X", "LB"]))
    assert set(names) == {"A", "X", "LB"}


def test_pad_state_helpers():
    s = PadState(lx=3.0 / 3.0, ly=-1.0)
    assert s.left_mag == pytest.approx(math.sqrt(2.0))
    assert PadState().neutral()


def make_clip(n=100, rate=250.0):
    clip = Clip(meta={"name": "t"})
    for i in range(n):
        t = i / rate
        clip.append(
            t,
            PadState(
                lx=math.cos(t * 3),
                ly=math.sin(t * 3),
                lt=min(1.0, t),
                btns=1 if 0.1 < t < 0.2 else 0,
            ),
        )
    return clip


def test_save_load_json(tmp_path):
    clip = make_clip()
    path = str(tmp_path / "c.atk.json")
    clip.save(path)
    again = Clip.load(path)
    assert len(again) == len(clip)
    assert again.lx == pytest.approx(clip.lx, abs=1e-4)
    assert again.btns == clip.btns
    assert again.rate_hz == pytest.approx(250.0, rel=0.01)


def test_save_load_gzip(tmp_path):
    clip = make_clip()
    path = str(tmp_path / "c.atk.json.gz")
    clip.save(path)
    assert os.path.exists(path)
    again = Clip.load(path)
    assert len(again) == len(clip)


def test_file_is_valid_json(tmp_path):
    clip = make_clip()
    path = str(tmp_path / "c.atk.json")
    clip.save(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["format"] == "autotrack.clip"
    assert data["version"] == 1
    assert len(data["samples"]["t"]) == len(clip)


def test_slice_and_markers():
    clip = make_clip(500)
    clip.add_marker(1.0, "поворот")
    part = clip.slice(0.5, 1.5)
    assert part.duration == pytest.approx(1.0, abs=0.01)
    assert part.markers and part.markers[0][1] == "поворот"
    assert part.markers[0][0] == pytest.approx(0.5, abs=0.01)


def test_button_runs():
    clip = Clip()
    for i in range(50):
        clip.append(i / 100.0, PadState(btns=1 if 10 <= i < 20 else 0))
    runs = clip.button_runs(["A"])
    assert len(runs) == 1
    assert runs[0]["duration"] == pytest.approx(0.1, abs=0.02)


def test_concat():
    a, b = make_clip(50), make_clip(50)
    joined = concat([a, b], gap=0.1)
    assert joined.duration == pytest.approx(a.duration + b.duration + 0.1, abs=0.02)


def test_legacy_list_format(tmp_path):
    frames = [{"lx": 0.1, "ly": -0.2, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0, "btns": 0} for _ in range(5)]
    path = str(tmp_path / "legacy.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(frames, fh)
    clip = Clip.load(path)
    assert len(clip) == 5
    assert clip.lx[0] == pytest.approx(0.1)

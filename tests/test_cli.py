"""Командный интерфейс: демо-маршрут, отчёты, обработка."""

import os

import pytest

from autotrack.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])


def test_demo_report_info(tmp_path, capsys):
    path = str(tmp_path / "demo.atk.json")
    assert main(["demo", "--duration", "3", "--out", path]) == 0
    assert os.path.exists(path)
    out = capsys.readouterr().out
    assert "ПЛАВНОСТЬ" in out

    assert main(["info", path]) == 0
    assert "Клип" in capsys.readouterr().out

    assert main(["report", path, "--trim-idle", "0"]) == 0
    assert "СРАВНЕНИЕ" in capsys.readouterr().out


def test_process_writes_file(tmp_path, capsys):
    src = str(tmp_path / "d.atk.json")
    main(["demo", "--duration", "2", "--out", src])
    out = str(tmp_path / "d.proc.atk.json")
    assert main(["process", src, "--out", out]) == 0
    assert os.path.exists(out)
    assert "Обработка" in capsys.readouterr().out


def test_play_dry_run(tmp_path):
    src = str(tmp_path / "d.atk.json")
    main(["demo", "--duration", "2", "--out", src])
    code = main(
        [
            "play", src, "--loops", "1", "--pad", "dry", "--quiet",
            "--countdown", "0", "--no-hotkeys", "--telemetry", "--telemetry-hz", "250",
        ]
    )
    assert code == 0


def test_report_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        main(["report", str(tmp_path / "nope.atk.json")])


def test_unknown_command():
    with pytest.raises(SystemExit):
        main(["нет-такой-команды"])

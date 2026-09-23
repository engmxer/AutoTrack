"""Командный интерфейс AutoTrack.

Примеры::

    python -m autotrack pads                      # какие геймпады видит система
    python -m autotrack monitor --device 0        # живые значения осей (проверка)
    python -m autotrack record route1 --duration 60 --window
    python -m autotrack play route1.atk.json --loops 5 --countdown 3
    python -m autotrack report route1.atk.json
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import List, Optional

from .frame import PadState, mask_names
from .metrics import analyze, compare_reports, format_report
from .processing import ProcessConfig, describe_process, process
from .session import Clip


def _load_clip(path: str) -> Clip:
    """Загрузить клип по имени: точный путь, с расширением или поиск по маске."""
    candidates = [path]
    if not os.path.exists(path):
        candidates = []
        for pat in (path, path + ".atk.json", path + ".atk.json.gz", path + ".json"):
            candidates += glob.glob(pat)
    if not candidates:
        # поиск в папке clips/
        base = os.path.basename(path)
        for pat in (base, base + ".atk.json", f"*{base}*"):
            candidates += glob.glob(os.path.join("clips", pat))
    if not candidates:
        raise SystemExit(f"Клип не найден: {path}")
    return Clip.load(candidates[0])


def _raw_path(path: str) -> str:
    """Путь для сырой (необработанной) копии клипа."""
    if path.endswith(".atk.json.gz"):
        return path[: -len(".atk.json.gz")] + ".raw.atk.json.gz"
    if path.endswith(".atk.json"):
        return path[: -len(".atk.json")] + ".raw.atk.json"
    return path + ".raw.atk.json"


def _default_clip_path(name: str, *, gz: bool = False) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    folder = "clips"
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, safe + ".atk.json" + (".gz" if gz else ""))


def _process_cfg_from_args(a: argparse.Namespace, *, default_on: bool = True) -> Optional[ProcessConfig]:
    if getattr(a, "no_process", False):
        return None
    if not default_on and not getattr(a, "process", False):
        return None
    return ProcessConfig(
        median=not getattr(a, "no_median", False),
        lowpass_hz=getattr(a, "lowpass", 14.0),
        resample_hz=getattr(a, "resample", 120.0),
        deadzone=getattr(a, "deadzone", 0.0),
        trigger_expand=not getattr(a, "no_trigger_ramp", False),
        trim_idle=getattr(a, "trim_idle", 0.0),
        interp=getattr(a, "interp", "linear"),
    )


# ---------------------------------------------------------------------------
# команды
# ---------------------------------------------------------------------------


def cmd_pads(a: argparse.Namespace) -> int:
    from .recorder import list_pads

    print("Геймпады в системе:")
    for line in list_pads():
        print("  " + line)
    print()
    print("Для виртуального вывода нужен драйвер ViGEmBus + `pip install vgamepad`.")
    return 0


def cmd_monitor(a: argparse.Namespace) -> int:
    from .recorder import PadSource

    src = PadSource(a.device)
    print(f"Геймпад: [{src.index}] {src.name}")
    print("Двигайте стики/курки. Проверьте: вверх — ly < 0, вправо — lx > 0, неподвижно — 0.")
    print("Выход: Ctrl+C")
    try:
        while True:
            st = src.read()
            bars = ""
            for name in ("lx", "ly", "rx", "ry"):
                v = getattr(st, name)
                bars += f" {name}={v:+.3f}"
            bars += f" LT={st.lt:.2f} RT={st.rt:.2f}"
            print(f"\r{bars}  [{'+'.join(mask_names(st.btns)) or '-':<24}]", end="", flush=True)
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("\nГотово.")
    finally:
        src.close()
    return 0


def cmd_record(a: argparse.Namespace) -> int:
    from .recorder import RecConfig, Recorder
    from .player import ConsoleKeys  # noqa: F401  (источник клавиш)

    cfg = RecConfig(
        duration=a.duration,
        rate_hz=a.rate,
        device=a.device,
        countdown=a.countdown,
        window=not a.no_window,
        marker_button=a.marker_button,
        stop_button=a.stop_button,
        stop_combo=a.stop_combo,
        stick_deadzone=0.0,
        process=not a.no_process,
        process_cfg=_process_cfg_from_args(a, default_on=True),
        live=not a.quiet,
        quiet=a.quiet,
        trim_seconds=a.trim_seconds,
    )
    rec = Recorder(cfg)
    res = rec.record(keys=ConsoleKeys())
    if len(res.clip) < 3:
        print("Слишком короткая запись — ничего не сохранено.")
        return 2
    out = a.out or _default_clip_path(a.name, gz=a.gz)
    raw_out = _raw_path(out)
    res.raw.save(raw_out)
    res.clip.save(out)
    print()
    print(format_report(res.raw_report, title="ОТЧЁТ: СЫРАЯ ЗАПИСЬ"))
    print(compare_reports(res.raw_report, res.report, title="сырая → обработанная"))
    print(format_report(res.report, title="ОТЧЁТ: КЛИП ДЛЯ ПОВТОРА"))
    print(f"\nСохранено:\n  клип : {out}\n  сырьё: {raw_out}")
    print(f"Запуск повтора: python -m autotrack play \"{out}\" --loops 0 --countdown 3")
    return 0


def cmd_play(a: argparse.Namespace) -> int:
    from .player import ConsoleKeys, Player, PlayConfig, format_play_result

    clip = _load_clip(a.clip)
    print(f"Клип: {clip.name} | {clip.duration:.2f} с | {len(clip)} сэмплов | ~{clip.rate_hz:.0f} Гц")
    cfg = PlayConfig(
        rate_hz=a.rate,
        speed=a.speed,
        loops=a.loops,
        loop_gap=a.loop_gap,
        deadzone=a.deadzone,
        interp=a.interp,
        game_fps=a.game_fps,
        min_press_scale=a.min_press_scale,
        filter_mode=a.filter,
        smooth_hz=a.smooth_hz,
        min_cutoff=a.min_cutoff,
        beta=a.beta,
        beta_angle=a.beta_angle,
        precise=not a.coarse,
        lag_tolerance=a.lag_tolerance_ms / 1000.0,
        lead_in=a.lead_in,
        entry_ramp_ms=a.entry_ramp,
        exit_ramp_ms=a.exit_ramp,
        tail_hold_ms=a.tail_hold,
        countdown=a.countdown,
        pad_backend=a.pad,
        pad_deadzone=a.pad_deadzone,
        invert_y=not a.no_invert_y,
        telemetry=a.telemetry,
        telemetry_hz=a.telemetry_hz,
        priority=a.priority,
        mmcss=not a.no_mmcss,
        cpu=a.cpu,
        quiet=a.quiet,
        live=not a.quiet,
    )
    player = Player(clip, cfg, process_cfg=_process_cfg_from_args(a, default_on=False))
    if player.processed_stats:
        print(describe_process(player.processed_stats))
    result = player.run(hotkeys=not a.no_hotkeys)
    print(format_play_result(result))
    if result.report:
        print(format_report(result.report, title="ПЛАВНОСТЬ ФАКТИЧЕСКОГО ВЫВОДА"))
        if a.telemetry_out:
            result.telemetry.save(a.telemetry_out)
            print(f"Телеметрия сохранена: {a.telemetry_out}")
    return 0 if not result.aborted else 130


def cmd_report(a: argparse.Namespace) -> int:
    clip = _load_clip(a.clip)
    # телеметрия — это уже фактический вывод, а не запись: чистить и сглаживать
    # в ней нечего, иначе отчёт начнёт описывать саму обработку
    is_telemetry = str(clip.meta.get("name", "")).startswith("telemetry") or "runs" in clip.meta
    raw_report = analyze(clip)
    print(format_report(raw_report, title="ОТЧЁТ О ПЛАВНОСТИ (как записано)"))
    if is_telemetry and not a.no_process:
        print("Это телеметрия вывода (метрики и так по фактическому сигналу) — обработка пропущена.")
        return 0
    if not a.no_process:
        res = process(clip, _process_cfg_from_args(a, default_on=True), source_name=clip.name)
        print(describe_process(res.stats))
        processed_report = analyze(res.clip)
        print(compare_reports(raw_report, processed_report, title="сырая → обработанная"))
        print(format_report(processed_report, title="ОТЧЁТ О ПЛАВНОСТИ (после сглаживания)"))
    return 0


def cmd_process(a: argparse.Namespace) -> int:
    clip = _load_clip(a.clip)
    res = process(clip, _process_cfg_from_args(a, default_on=True), source_name=clip.name)
    out = a.out or (os.path.splitext(a.clip)[0] + ".processed.atk.json")
    res.clip.save(out)
    print(describe_process(res.stats))
    print(f"Сохранено: {out}")
    return 0


def cmd_info(a: argparse.Namespace) -> int:
    clip = _load_clip(a.clip)
    for line in clip.summary_lines():
        print(line)
    if clip.markers:
        print("Метки:")
        for t, label in clip.markers:
            print(f"  {t:7.2f} с  {label}")
    runs = clip.button_runs()
    if runs:
        print(f"Нажатий всего: {len(runs)}")
    return 0


def cmd_demo(a: argparse.Namespace) -> int:
    """Сгенерировать демонстрационный маршрут (проверка настройки без руля)."""
    import math
    import random

    random.seed(42)
    clip = Clip(meta={"name": "demo", "device": "synthetic"})
    dur = a.duration
    rate = a.rate
    n = int(dur * rate)
    for i in range(n):
        t = i / rate
        # «змейка» + круг: газ и руль
        ph = t * 1.2
        lx = 0.55 * math.sin(ph * 1.7) + 0.15 * math.sin(ph * 5.0)
        ly = 0.45 * math.sin(ph) - 0.10
        rt = max(0.0, min(1.0, 0.5 + 0.5 * math.sin(ph * 0.8)))
        btns = 0
        if int(t) % 3 == 1 and (t % 1.0) < 0.15:
            btns |= 1 << 0  # A
        clip.append(t, PadState(lx=lx, ly=ly, lt=0.0, rt=rt, btns=btns))
    out = a.out or _default_clip_path("demo")
    clip.save(out)
    print(f"Демо-клип создан: {out} ({clip.duration:.1f} с)")
    print(format_report(analyze(clip), title="ДЕМО: ПЛАВНОСТЬ"))
    print(f"Проверка повтора: python -m autotrack play \"{out}\" --loops 2 --pad dry --quiet")
    return 0


def cmd_selftest(a: argparse.Namespace) -> int:
    """Быстрая самопроверка: тайминги, фильтры, метрики, сверка вывода."""
    import math

    from .frame import PadState
    from .player import PlayConfig, Player, format_play_result
    from .timing import DeadlineLoop, TimerResolution, freeze_gc

    ok = True
    print("1) Точность планировщика 250 Гц (2 попытки по 2 с, берём лучшую)…")
    best: dict = {}
    best_report = ""
    with TimerResolution(1):
        # прогрев: в свежем процессе первый прогон всегда хуже
        # (ленивые импорты, аллокатор, настройка приоритетов) — он не показателен
        warm = DeadlineLoop(250.0, spin_margin=0.00025)
        warm.start()
        with freeze_gc():
            for k in range(250):
                warm.wait_for(k)
        # вторая попытка нужна из-за внешних «заиканий» системы (обновления,
        # антивирус, чужой процесс): один провал ещё не значит, что виноват скрипт
        for _ in range(2):
            loop = DeadlineLoop(250.0, spin_margin=0.00025)
            loop.start()
            with freeze_gc():
                for k in range(500):
                    loop.wait_for(k)
            summary = loop.stats.summary()
            if not best or summary["missed_deadlines"] < best["missed_deadlines"]:
                best, best_report = summary, loop.stats.report()
    print("   " + best_report)
    if best["missed_deadlines"] > 2:
        print("   ⚠ пропущены сроки — проверьте загрузку системы")
        ok = False

    print("2) Воспроизведение круга в «сухом» режиме (2 круга)…")
    clip = Clip(meta={"name": "selftest"})
    revolutions = 2
    dur = 2.0
    for i in range(int(dur * 250)):
        t = i / 250.0
        ph = 2.0 * math.pi * revolutions * t / dur
        clip.append(t, PadState(lx=0.6 * math.cos(ph), ly=0.6 * math.sin(ph)))
    cfg = PlayConfig(
        rate_hz=250, loops=2, countdown=0, quiet=True, pad_backend="dry", live=False,
        telemetry=True, telemetry_hz=250,
    )
    player = Player(clip, cfg)
    result = player.run(hotkeys=False)
    print(format_play_result(result, title="САМОПРОВЕРКА"))
    if result.loop_stats.get("missed_deadlines", 0) > 2:
        ok = False
    score = result.report.get("smoothness_score", 0)
    st = (result.report.get("sticks") or {}).get("L", {})
    print(
        f"   Плавность вывода: {score}/100  (дрожание {st.get('noise_pct', 0):.2f}% полной шкалы, "
        f"угловое ускорение p95 {st.get('angular_accel_p95', 0):.1f} рад/с²)"
    )
    if score < 60:
        print("   ⚠ вывод неровный — проверьте загрузку CPU (анализ, браузер, запись экрана)")
        ok = False
    missed_btn = result.verification.get("buttons_missed", 0)
    if missed_btn:
        ok = False
    print(
        f"   Проверки: пропущенных сроков {result.loop_stats.get('missed_deadlines', 0)} (норма ≤2), "
        f"оценка {score:.1f} (норма ≥60), потерянных нажатий {missed_btn}"
    )
    print("\nИТОГ: " + ("всё в порядке ✅" if ok else "есть замечания ⚠"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# разбор аргументов
# ---------------------------------------------------------------------------


def _add_process_args(p: argparse.ArgumentParser, *, default_on: bool) -> None:
    g = p.add_argument_group("обработка записи")
    g.add_argument("--no-process", action="store_true", help="не обрабатывать (играть как записано)")
    g.add_argument("--process", action="store_true", help="обработать клип перед повтором")
    g.add_argument("--lowpass", type=float, default=14.0, help="срез ФНЧ, Гц (0 = выключить)")
    g.add_argument("--resample", type=float, default=120.0, help="частота ровной сетки, Гц (0 = не менять)")
    g.add_argument("--no-median", action="store_true", help="не убирать одиночные выбросы")
    g.add_argument("--no-trigger-ramp", action="store_true", help="не разглаживать курки 0/1")
    g.add_argument("--trim-idle", type=float, default=0.25, help="обрезать простой в начале/конце, с")


def _add_play_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("воспроизведение")
    g.add_argument("--rate", type=float, default=250.0, help="частота вывода в геймпад, Гц (по умолчанию 250)")
    g.add_argument("--speed", type=float, default=1.0, help="множитель скорости (0.5 — медленнее)")
    g.add_argument("--loops", type=int, default=1, help="кругов (0 = бесконечно)")
    g.add_argument("--loop-gap", type=float, default=0.15, help="пауза между кругами, с")
    g.add_argument("--countdown", type=float, default=3.0, help="отсчёт перед стартом, с")
    g.add_argument("--lead-in", type=float, default=0.0, help="нейтраль перед стартом, с")
    g.add_argument("--entry-ramp", type=float, default=120.0, help="плавный вход из нейтрали, мс")
    g.add_argument("--exit-ramp", type=float, default=120.0, help="плавный выход в нейтраль, мс")
    g.add_argument("--tail-hold", type=float, default=30.0, help="удержание нейтрали в конце, мс")
    g.add_argument("--deadzone", type=float, default=0.0, help="мёртвая зона стиков плана (чистка записи)")
    g.add_argument("--interp", choices=["linear", "cubic", "hold"], default="linear", help="интерполяция")
    g.add_argument("--game-fps", type=float, default=60.0, help="кадров/с в игре (для минимального нажатия)")
    g.add_argument("--min-press-scale", type=float, default=1.5, help="во сколько раз удлинять короткие нажатия")
    g.add_argument(
        "--filter", choices=["off", "zero", "polar", "xy"], default="zero",
        help="сглаживание: zero — без задержки (по умолчанию), polar/xy — «на лету»",
    )
    g.add_argument("--smooth-hz", type=float, default=15.0, help="срез сглаживания маршрута, Гц (0 = выключить)")
    g.add_argument("--min-cutoff", type=float, default=1.6, help="One Euro: срез в покое, Гц (меньше = плавнее)")
    g.add_argument("--beta", type=float, default=0.02, help="One Euro: реакция на скорость")
    g.add_argument("--beta-angle", type=float, default=0.012, help="One Euro: реакция для угла стика")
    g.add_argument("--pad", default="xinput", help="вывод: xinput | dry | null")
    g.add_argument("--pad-deadzone", type=float, default=0.0, help="мёртвая зона на самом виртуальном геймпаде")
    g.add_argument("--no-invert-y", action="store_true", help="не инвертировать Y (если стик в игре «наоборот»)")
    g.add_argument("--coarse", action="store_true", help="грубый таймер (без спин-ожидания)")
    g.add_argument("--lag-tolerance-ms", type=float, default=2.0, help="допустимое опоздание кадра, мс")
    g.add_argument("--priority", choices=["normal", "above_normal", "high", "realtime"], default="high")
    g.add_argument("--no-mmcss", action="store_true", help="не регистрировать поток в MMCSS «Games»")
    g.add_argument("--cpu", type=int, default=None, help="привязать поток к ядру (например 3)")
    g.add_argument("--telemetry", action="store_true", help="записать фактический вывод и посчитать метрики")
    g.add_argument("--telemetry-hz", type=float, default=120.0, help="частота телеметрии, Гц")
    g.add_argument("--telemetry-out", default="", help="куда сохранить телеметрию")
    g.add_argument("--quiet", action="store_true", help="без живых строк вывод")
    g.add_argument("--no-hotkeys", action="store_true", help="не читать клавиши консоли")
    _add_process_args(p, default_on=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autotrack",
        description="AutoTrack — запись и воспроизведение маршрута (The Track) на виртуальном геймпаде.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="version", version="AutoTrack 0.1.0")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("pads", help="список геймпадов")
    s.set_defaults(func=cmd_pads)

    s = sub.add_parser("monitor", help="живые значения осей (проверка знаков)")
    s.add_argument("--device", type=int, default=None)
    s.set_defaults(func=cmd_monitor)

    s = sub.add_parser("record", help="записать маршрут")
    s.add_argument("name", nargs="?", default="route", help="имя маршрута")
    s.add_argument("--duration", type=float, default=0.0, help="секунд записи (0 = до остановки)")
    s.add_argument("--rate", type=float, default=250.0, help="частота записи, Гц")
    s.add_argument("--device", type=int, default=None, help="индекс геймпада")
    s.add_argument("--countdown", type=float, default=3.0)
    s.add_argument("--window", dest="no_window", action="store_false", help="показать окно (по умолчанию да)")
    s.add_argument("--no-window", dest="no_window", action="store_true", help="без окна")
    s.add_argument("--marker-button", default="", help="кнопка для метки (например Y)")
    s.add_argument("--stop-button", default="", help="кнопка остановки (например BACK)")
    s.add_argument("--stop-combo", default="", help="комбинация остановки (например BACK+START)")
    s.add_argument("--out", default="", help="файл клипа")
    s.add_argument("--gz", action="store_true", help="сжать клип (gzip)")
    s.add_argument("--trim-seconds", type=float, default=0.0, help="отрезать N секунд с конца")
    s.add_argument("--quiet", action="store_true")
    _add_process_args(s, default_on=True)
    s.set_defaults(func=cmd_record, no_process=False, no_median=False, no_trigger_ramp=False)

    s = sub.add_parser("play", help="воспроизвести маршрут", parents=[])
    s.add_argument("clip", help="файл клипа")
    _add_play_args(s)
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("report", help="отчёт о плавности клипа")
    s.add_argument("clip")
    s.add_argument("--no-process", action="store_true", help="без сравнения со сглаженным")
    s.add_argument("--lowpass", type=float, default=14.0)
    s.add_argument("--resample", type=float, default=120.0)
    s.add_argument("--no-median", action="store_true")
    s.add_argument("--no-trigger-ramp", action="store_true")
    s.add_argument("--trim-idle", type=float, default=0.0)
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("process", help="сгладить клип и сохранить")
    s.add_argument("clip")
    s.add_argument("--out", default="")
    _add_process_args(s, default_on=True)
    s.set_defaults(func=cmd_process, no_process=False, no_median=False, no_trigger_ramp=False)

    s = sub.add_parser("info", help="краткая информация о клипе")
    s.add_argument("clip")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("demo", help="создать демонстрационный клип")
    s.add_argument("--duration", type=float, default=15.0)
    s.add_argument("--rate", type=float, default=120.0)
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("selftest", help="самопроверка таймингов и пайплайна")
    s.set_defaults(func=cmd_selftest)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # аргументы, которых нет у конкретной команды
    for name, default in (
        ("no_process", False),
        ("no_median", False),
        ("no_trigger_ramp", False),
        ("trim_idle", 0.0),
        ("lowpass", 14.0),
        ("resample", 120.0),
        ("deadzone", 0.0),
        ("interp", "linear"),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)
    try:
        return int(args.func(args) or 0)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nПрервано.")
        return 130
    except Exception as exc:  # понятная ошибка вместо трассировки
        print(f"\nОшибка: {exc}", file=sys.stderr)
        if os.environ.get("AUTOTRACK_DEBUG"):
            raise
        return 1


__all__ = ["main", "build_parser"]

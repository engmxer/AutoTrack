"""Формат клипа AutoTrack (записанная последовательность действий).

Файл — обычный JSON (при желании сжатый gzip), расширение ``*.atk.json`` или
``*.atk.json.gz``. Числа осей хранятся целыми (``значение * 100000``), поэтому
файл компактный, а точность — 1/100000 полной шкалы (заметно выше, чем
разрешение XInput 1/32767, которое всё равно увидит игра).

Структура::

    {
      "format": "autotrack.clip",
      "version": 1,
      "meta": {...},
      "samples": {
         "t":   [0.0, 0.004, ...],        # секунды от начала, монотонно
         "lx":  [0, -1234, ...],          # -100000..100000
         "ly":  [...], "rx": [...], "ry": [...],
         "lt":  [...], "rt": [...],       # 0..10000
         "btns":[0, 0, 32, ...]           # битовая маска
      },
      "markers": [[1.23, "поворот 1"], ...]
    }
"""

from __future__ import annotations

import gzip
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .frame import AXES, STICK_AXES, TRIGGER_AXES, PadState

FORMAT = "autotrack.clip"
VERSION = 1

#: Множитель для хранения вещественных осей целыми числами.
SCALE = 100000.0


def _round_t(v: float) -> float:
    return round(float(v), 5)


def _norm(v: float) -> float:
    """Ось стика: -1..1 с округлением до 1/100000 (точность записи)."""
    return round(max(-1.0, min(1.0, float(v))), 5)


def _norm01(v: float) -> float:
    """Курок: 0..1 с округлением до 1/100000."""
    return round(max(0.0, min(1.0, float(v))), 5)


def _to_int(v: float) -> int:
    return int(round(float(v) * SCALE))


@dataclass
class Clip:
    """Записанные (или сгенерированные) данные геймпада с таймстампами."""

    t: List[float] = field(default_factory=list)
    lx: List[float] = field(default_factory=list)
    ly: List[float] = field(default_factory=list)
    rx: List[float] = field(default_factory=list)
    ry: List[float] = field(default_factory=list)
    lt: List[float] = field(default_factory=list)
    rt: List[float] = field(default_factory=list)
    btns: List[int] = field(default_factory=list)
    markers: List[Tuple[float, str]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None

    # -- базовые свойства ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return self.t[-1] if self.t else 0.0

    @property
    def rate_hz(self) -> float:
        """Оценка частоты дискретизации (медиана по dt)."""
        if len(self.t) < 3:
            return float(self.meta.get("rate_hz") or 0.0)
        dts = [b - a for a, b in zip(self.t, self.t[1:]) if b > a]
        if not dts:
            return 0.0
        return 1.0 / statistics.median(dts)

    @property
    def name(self) -> str:
        return self.meta.get("name") or (os.path.basename(self.path) if self.path else "clip")

    def axis(self, name: str) -> List[float]:
        return getattr(self, name)

    # -- доступ -------------------------------------------------------------
    def frame(self, i: int) -> PadState:
        return PadState(
            lx=self.lx[i], ly=self.ly[i], rx=self.rx[i], ry=self.ry[i],
            lt=self.lt[i], rt=self.rt[i], btns=self.btns[i],
        )

    def append(self, t: float, s: PadState) -> None:
        """Добавить кадр (значения нормализуются и квантуются, как в файле)."""
        self.t.append(_round_t(t))
        self.lx.append(_norm(s.lx))
        self.ly.append(_norm(s.ly))
        self.rx.append(_norm(s.rx))
        self.ry.append(_norm(s.ry))
        self.lt.append(_norm01(s.lt))
        self.rt.append(_norm01(s.rt))
        self.btns.append(int(s.btns))

    def append_raw(self, t: float, s: PadState) -> None:
        """Добавить кадр без нормализации (для джойнов/генерации планов)."""
        self.t.append(float(t))
        self.lx.append(float(s.lx))
        self.ly.append(float(s.ly))
        self.rx.append(float(s.rx))
        self.ry.append(float(s.ry))
        self.lt.append(float(s.lt))
        self.rt.append(float(s.rt))
        self.btns.append(int(s.btns))

    def add_marker(self, t: float, label: str = "") -> None:
        self.markers.append((_round_t(t), label))

    # -- преобразования -----------------------------------------------------
    def slice(self, t0: float = 0.0, t1: Optional[float] = None) -> "Clip":
        """Кусок [t0, t1] с сохранением относительного времени (сдвиг к нулю)."""
        t1 = self.duration if t1 is None else t1
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        out = Clip(meta=dict(self.meta))
        out.meta["source"] = self.name
        out.meta["slice"] = [lo, hi]
        for i, ti in enumerate(self.t):
            if lo - 1e-9 <= ti <= hi + 1e-9:
                out.append_raw(ti - lo, self.frame(i))
        if not out.t and self.t:
            out.append_raw(0.0, self.frame(0))
        out.markers = [(max(0.0, mt - lo), lbl) for mt, lbl in self.markers if lo <= mt <= hi]
        return out

    def reindex(self) -> None:
        """Проставить ``t`` как кумулятивную сумму dt (0..duration) и проверить порядок."""
        self.t.sort()
        self.meta["duration"] = self.duration

    def channel_names(self) -> List[str]:
        return list(AXES)

    # -- сохранение/загрузка ------------------------------------------------
    def to_dict(self, *, compact: bool = True) -> Dict[str, Any]:
        meta = dict(self.meta)
        meta.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        meta.setdefault("duration", round(self.duration, 5))
        meta.setdefault("rate_hz", round(self.rate_hz, 3))
        meta.setdefault("samples", len(self.t))
        data: Dict[str, Any] = {
            "format": FORMAT,
            "version": VERSION,
            "meta": meta,
            "samples": {
                "t": [_round_t(x) for x in self.t],
                "lx": [_to_int(x) for x in self.lx],
                "ly": [_to_int(x) for x in self.ly],
                "rx": [_to_int(x) for x in self.rx],
                "ry": [_to_int(x) for x in self.ry],
                "lt": [_to_int(x) for x in self.lt],
                "rt": [_to_int(x) for x in self.rt],
                "btns": [int(x) for x in self.btns],
            },
        }
        if self.markers:
            data["markers"] = [[_round_t(t), lbl] for t, lbl in self.markers]
        return data

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, separators=(",", ":") if indent is None else None)

    def save(self, path: str, *, indent: Optional[int] = None) -> str:
        """Сохранить клип. Расширение ``.gz`` включает сжатие."""
        path = self._ensure_ext(path)
        text = self.to_json(indent=indent)
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        if path.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                fh.write(text)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        self.path = path
        self.meta.setdefault("name", os.path.basename(path))
        return path

    @staticmethod
    def _ensure_ext(path: str) -> str:
        if path.endswith(".gz") or path.endswith(".json") or path.endswith(".atk"):
            return path
        return path + ".atk.json"

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: Optional[str] = None) -> "Clip":
        fmt = data.get("format")
        if fmt != FORMAT:
            raise ValueError(f"неизвестный формат файла: {fmt!r} (ожидался {FORMAT!r})")
        if int(data.get("version", 0)) > VERSION:
            raise ValueError(
                f"файл создан более новой версией AutoTrack (version={data.get('version')})"
            )
        s = data.get("samples") or {}
        n = len(s.get("t", []))
        clip = cls(meta=dict(data.get("meta") or {}), path=path)
        clip.t = [float(x) for x in s.get("t", [])]
        clip.lx = [int(x) / SCALE for x in s.get("lx", [0] * n)]
        clip.ly = [int(x) / SCALE for x in s.get("ly", [0] * n)]
        clip.rx = [int(x) / SCALE for x in s.get("rx", [0] * n)]
        clip.ry = [int(x) / SCALE for x in s.get("ry", [0] * n)]
        clip.lt = [int(x) / SCALE for x in s.get("lt", [0] * n)]
        clip.rt = [int(x) / SCALE for x in s.get("rt", [0] * n)]
        clip.btns = [int(x) for x in s.get("btns", [0] * n)]
        clip.markers = [(float(t), str(lbl)) for t, lbl in (data.get("markers") or [])]
        return clip

    @classmethod
    def from_json(cls, text: str, path: Optional[str] = None) -> "Clip":
        return cls.from_dict(json.loads(text), path=path)

    @classmethod
    def load(cls, path: str) -> "Clip":
        """Загрузить клип (поддерживает gzip и «голый» список, и обычный JSON)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"файл не найден: {path}")
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        if isinstance(data, list):  # совместимость: массив кадров
            clips = [PadState(**f) for f in data]
            out = cls(meta={"name": os.path.basename(path)})
            for i, st in enumerate(clips):
                out.append(i / 60.0, st)
            return out
        return cls.from_dict(data, path=path)

    # -- статистика ---------------------------------------------------------
    def button_runs(self, names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """Нажатия кнопок как интервалы [start, end, длительность, имя]."""
        from .frame import BUTTON_BITS

        selected = {n: 1 << BUTTON_BITS[n] for n in (names or list(BUTTON_BITS)) if n in BUTTON_BITS}
        runs: List[Dict[str, Any]] = []
        if not self.t:
            return runs
        state: Dict[str, Optional[float]] = {}
        for i, t in enumerate(self.t):
            b = self.btns[i]
            for name, m in selected.items():
                pressed = bool(b & m)
                started = state.get(name)
                if pressed and started is None:
                    state[name] = t
                elif not pressed and started is not None:
                    runs.append({"button": name, "start": started, "end": t, "duration": t - started})
                    state[name] = None
        end_t = self.duration
        for name, started in state.items():
            if started is not None:
                runs.append({"button": name, "start": started, "end": end_t, "duration": end_t - started})
        runs.sort(key=lambda r: r["start"])
        return runs

    def quick_stats(self) -> Dict[str, Any]:
        """Краткая сводка: сколько времени стики в движении, где центр и т.д."""
        n = len(self.t) or 1
        out: Dict[str, Any] = {
            "samples": len(self.t),
            "duration": round(self.duration, 3),
            "rate_hz": round(self.rate_hz, 2),
            "buttons_pressed": len(self.button_runs()),
        }
        for side, (ax, ay) in STICK_AXES.items():
            mags = [math.hypot(self.axis(ax)[i], self.axis(ay)[i]) for i in range(len(self.t))]
            moving = [m for m in mags if m > 0.08]
            out[f"stick_{side.lower()}_moving_pct"] = round(100.0 * len(moving) / n, 1)
            out[f"stick_{side.lower()}_mean_mag"] = round(sum(mags) / n, 4)
            out[f"stick_{side.lower()}_max_mag"] = round(max(mags, default=0.0), 4)
        for ax in TRIGGER_AXES:
            vals = self.axis(ax)
            out[f"{ax}_active_pct"] = round(100.0 * sum(1 for v in vals if v > 0.05) / n, 1)
        return out

    def summary_lines(self) -> List[str]:
        s = self.quick_stats()
        return [
            f"Клип: {self.name}",
            f"Длительность: {s['duration']} с, сэмплов: {s['samples']}, частота: {s['rate_hz']} Гц",
            f"Нажатий кнопок: {s['buttons_pressed']}",
            f"Левый стик в движении: {s['stick_l_moving_pct']}% времени",
            f"Правый стик в движении: {s['stick_r_moving_pct']}% времени",
        ]


def concat(clips: Iterable[Clip], *, gap: float = 0.0) -> Clip:
    """Склеить клипы последовательно (с необязательной паузой между ними)."""
    out = Clip(meta={"name": "concat"})
    offset = 0.0
    for idx, c in enumerate(clips):
        if idx and gap:
            out.append_raw(offset + gap, PadState())
            offset += gap
        for i in range(len(c)):
            out.append_raw(offset + c.t[i], c.frame(i))
        offset += c.duration
        for mt, lbl in c.markers:
            out.add_marker(mt + (offset - c.duration), lbl)
    out.meta["duration"] = out.duration
    return out

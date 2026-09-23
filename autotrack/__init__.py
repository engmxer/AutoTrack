"""AutoTrack — запись и воспроизведение маршрута джойстика (NBA 2K, The Track).

Что делает пакет
----------------
1. **Запись** (``autotrack.recorder``): слушает физический геймпад с точной
   частотой (по умолчанию 250 Гц) и сохраняет клип.
2. **Обработка** (``autotrack.processing``): нуль-фазовое сглаживание,
   удаление выбросов, ровная сетка времени, разглаживание курков.
3. **Воспроизведение** (``autotrack.player``): выдаёт маршрут в виртуальный
   Xbox-геймпад (``vgamepad`` + ViGEmBus) с интерполяцией, сглаживанием
   «на лету» и вытеснением задержек (``autotrack.timing``).
4. **Метрики** (``autotrack.metrics``): плавность стиков — дрожание руки,
   неровность хода, угловая скорость/ускорение, дрожание направления,
   устойчивые развороты, сводная оценка 0..100.

Быстрый старт::

    python -m autotrack pads
    python -m autotrack record route1 --duration 60
    python -m autotrack play clips/route1.atk.json --loops 0
    python -m autotrack report clips/route1.atk.json
"""

__version__ = "0.1.0"

from .frame import PadState  # noqa: F401
from .session import Clip  # noqa: F401

__all__ = ["PadState", "Clip", "__version__"]

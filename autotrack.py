#!/usr/bin/env python3
"""Удобный запуск без установки пакета: ``python autotrack.py <команда>``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autotrack.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

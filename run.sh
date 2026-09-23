#!/usr/bin/env bash
# AutoTrack: быстрый запуск (Linux/macOS — запись и отчёты; виртуальный
# геймпад доступен только на Windows).
set -e
cd "$(dirname "$0")"
case "${1:-menu}" in
  record)  shift; exec python3 autotrack.py record "$@" ;;
  play)    shift; exec python3 autotrack.py play "$@" ;;
  report)  shift; exec python3 autotrack.py report "$@" ;;
  demo)    shift; exec python3 autotrack.py demo "$@" ;;
  pads)    exec python3 autotrack.py pads ;;
  monitor) exec python3 autotrack.py monitor ;;
  test)    exec python3 -m pytest -q ;;
  selftest) exec python3 autotrack.py selftest ;;
  *)
    echo "Использование: ./run.sh {record|play|report|demo|pads|monitor|test|selftest} [аргументы]"
    echo "Например: ./run.sh record track1 --duration 60"
    ;;
esac

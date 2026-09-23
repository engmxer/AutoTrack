@echo off
chcp 65001 >nul
title AutoTrack - прокат маршрута в The Track
cd /d "%~dp0"

:menu
cls
echo ============================================================
echo   AutoTrack - запись и повтор маршрута (The Track)
echo ============================================================
echo.
echo   [1] Записать маршрут      (нужен подключённый геймпад)
echo   [2] Повторить маршрут     (нужен драйвер ViGEmBus)
echo   [3] Отчёт о плавности
echo   [4] Список геймпадов
echo   [5] Проверка осей (monitor)
echo   [6] Самопроверка таймингов (без игры)
echo   [7] Демо-маршрут (проверка без геймпада)
echo   [8] Обработать запись (сгладить и сохранить)
echo   [0] Выход
echo.
set /p choice="Выбор: "

if "%choice%"=="1" goto record
if "%choice%"=="2" goto play
if "%choice%"=="3" goto report
if "%choice%"=="4" goto pads
if "%choice%"=="5" goto monitor
if "%choice%"=="6" goto selftest
if "%choice%"=="7" goto demo
if "%choice%"=="8" goto process
if "%choice%"=="0" exit /b
goto menu

:record
set /p name="Имя маршрута (например track1): "
set /p dur="Длительность в секундах (0 = до кнопки стоп): "
python autotrack.py record "%name%" --duration %dur% --countdown 3
pause
goto menu

:play
set /p file="Файл клипа (например clips\track1.atk.json): "
set /p loops="Сколько кругов (0 = бесконечно): "
echo.
echo   Сглаживание маршрута (нуль-фазовое, без задержки):
echo     15 - по умолчанию: ровно, быстрые флики проходят целиком
echo      8 - мягче (сильнее гасит дрожь руки)
echo     30 - почти как записано
echo      0 - выключить сглаживание
set /p smooth="Срез, Гц (Enter = 15): "
if "%smooth%"=="" set smooth=15
python autotrack.py play "%file%" --loops %loops% --countdown 3 --telemetry --smooth-hz %smooth%
pause
goto menu

:process
set /p file="Файл клипа: "
set /p out="Куда сохранить (Enter = рядом, имя + .processed.atk.json): "
if "%out%"=="" (python autotrack.py process "%file%" --lowpass 10) else (python autotrack.py process "%file%" --lowpass 10 --out "%out%")
pause
goto menu

:report
set /p file="Файл клипа: "
python autotrack.py report "%file%"
pause
goto menu

:pads
python autotrack.py pads
pause
goto menu

:monitor
python autotrack.py monitor
pause
goto menu

:selftest
python autotrack.py selftest
pause
goto menu

:demo
python autotrack.py demo --duration 15
pause
goto menu

@echo off
REM ---------------------------------------------------------------------------
REM Rebuild the FluxRT kiosk runtime (venv + model weights) from scratch.
REM Double-click this after a teardown, or on a fresh Windows box with a recent
REM NVIDIA driver + winget. Delegates to setup_kiosk.ps1.
REM ---------------------------------------------------------------------------
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_kiosk.ps1"
echo.
pause

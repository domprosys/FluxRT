@echo off
setlocal
REM ===========================================================================
REM FluxRT production launcher (kiosk).
REM Double-click the "FluxRT" desktop shortcut, which points here.
REM
REM - Runs from the repo root so the relative model/config paths resolve.
REM - Initializes MSVC (via run_msvc.bat) so torch.compile works.
REM - Persists the compiled-kernel caches inside the repo so torch.compile
REM   warms up fast on EVERY launch, not just the first one ever.
REM - Uses configs/production_config.json (cached prompt embeds, alpha 0.3).
REM ===========================================================================
cd /d "%~dp0.."

set "TORCHINDUCTOR_CACHE_DIR=%CD%\.cache\torchinductor"
set "TRITON_CACHE_DIR=%CD%\.cache\triton"
if not exist ".cache" mkdir ".cache"

echo.
echo  ==========================================================
echo   Starting FluxRT...
echo.
echo   The FIRST launch warms up (kernel compile + prompt
echo   encode) and may take 1-2 minutes. Later launches are
echo   much faster.
echo.
echo   A window will open. Move it to the second screen and
echo   press F11 for fullscreen. Press Esc to exit.
echo  ==========================================================
echo.

call "%~dp0run_msvc.bat" scripts/run_gui.py --int8 --config configs/production_config.json

if errorlevel 1 (
  echo.
  echo  FluxRT exited with an error. Take a photo/screenshot of this window.
  pause
)
endlocal

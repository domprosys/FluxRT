@echo off
REM ---------------------------------------------------------------------------
REM Initialize the MSVC build environment (cl.exe + INCLUDE/LIB) so PyTorch's
REM TorchInductor backend can compile C++/CUDA kernels when compile_models=true,
REM then launch the project's venv Python with whatever args are passed.
REM
REM Usage:
REM   scripts\run_msvc.bat scripts/run_gui.py --int8 --config configs/<cfg>.json
REM ---------------------------------------------------------------------------
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo [run_msvc] ERROR: failed to initialize MSVC environment (vcvars64.bat^).
  exit /b 1
)
"%~dp0..\.venv\Scripts\python.exe" %*

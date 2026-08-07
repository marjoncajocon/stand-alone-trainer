@echo off
REM Full rebuild for Windows and Linux, plus vet and tests.
REM CPU-only by design; the CUDA build happens on Colab (colab\build_colab.sh).
powershell -ExecutionPolicy Bypass -File "%~dp0build_all.ps1" -Force -Test
pause

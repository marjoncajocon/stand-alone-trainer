@echo off
REM Incremental rebuild - skips targets whose source hash is unchanged.
powershell -ExecutionPolicy Bypass -File "%~dp0build_all.ps1"
pause

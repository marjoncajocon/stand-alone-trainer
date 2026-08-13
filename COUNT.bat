@echo off
REM Data inventory: total games and total positions per variant.
REM
REM Same role as the engine kits' c_port\tools\trainer\COUNT.bat, but it runs the
REM count binary instead of a PowerShell script, so the numbers come from the
REM trainer's own loader rules rather than from grepping .log files.
REM
REM Built by BUILD.bat / UPDATE.bat alongside the trainer.
if not exist "%~dp0count_win_x64.exe" (
  echo count_win_x64.exe not built yet - run UPDATE.bat first.
  pause
  exit /b 1
)
"%~dp0count_win_x64.exe" --files
pause

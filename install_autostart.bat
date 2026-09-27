@echo off
setlocal

:: ================================================================
::  DeepRetail — Auto-Start Installer
::  Registers two Task Scheduler jobs that start automatically
::  every time you log in to Windows — no manual bat file needed.
::
::  Run this ONCE as Administrator, then reboot to verify.
:: ================================================================

set "PROJ=%~dp0"
set "PROJ=%PROJ:~0,-1%"
set "PYTHON=python"

echo.
echo  DeepRetail Auto-Start Installer
echo  Project dir: %PROJ%
echo.

:: --- Task 1: FastAPI Backend (always-on) -------------------------
echo [1/2] Registering DeepRetail Backend...
schtasks /Delete /TN "DeepRetail\Backend" /F >nul 2>&1
schtasks /Create ^
  /TN "DeepRetail\Backend" ^
  /TR "\"%PYTHON%\" -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000" ^
  /SC ONLOGON ^
  /RU "%USERNAME%" ^
  /RP * ^
  /SD "01/01/2025" ^
  /ST 00:00 ^
  /F ^
  /RL HIGHEST ^
  /DELAY 0000:10
if %ERRORLEVEL% EQU 0 (
    echo    Backend task created OK
) else (
    echo    ERROR creating backend task. Re-run as Administrator.
    pause & exit /b 1
)

:: Wrap in VBScript so it runs hidden (no console window on login)
set "VBS=%PROJ%\launch_backend.vbs"
(
echo Set WShell = CreateObject^("WScript.Shell"^)
echo WShell.CurrentDirectory = "%PROJ%"
echo WShell.Run "cmd /c python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 >> logs\backend.log 2>&1", 0, False
) > "%VBS%"

schtasks /Delete /TN "DeepRetail\Backend" /F >nul 2>&1
schtasks /Create ^
  /TN "DeepRetail\Backend" ^
  /TR "wscript.exe \"%VBS%\"" ^
  /SC ONLOGON ^
  /RU "%USERNAME%" ^
  /F ^
  /RL HIGHEST ^
  /DELAY 0000:05
echo    Backend auto-start: ON LOGON (hidden window)

:: --- Task 2: AI Pipeline (live detection) ------------------------
echo.
echo [2/2] Registering DeepRetail Pipeline...

set "VBS2=%PROJ%\launch_pipeline.vbs"
(
echo Set WShell = CreateObject^("WScript.Shell"^)
echo WShell.CurrentDirectory = "%PROJ%"
echo WShell.Run "cmd /c python run_demo.py --reset --anomaly --source 0 --no-display >> logs\pipeline.log 2>&1", 0, False
) > "%VBS2%"

schtasks /Delete /TN "DeepRetail\Pipeline" /F >nul 2>&1
schtasks /Create ^
  /TN "DeepRetail\Pipeline" ^
  /TR "wscript.exe \"%VBS2%\"" ^
  /SC ONLOGON ^
  /RU "%USERNAME%" ^
  /F ^
  /RL HIGHEST ^
  /DELAY 0000:20
echo    Pipeline auto-start: ON LOGON (20 sec delay after backend)

:: --- Done --------------------------------------------------------
echo.
echo  ================================================================
echo   Auto-start installed successfully!
echo.
echo   Both tasks start automatically when you log in to Windows.
echo   Logs are written to:
echo     %PROJ%\logs\backend.log
echo     %PROJ%\logs\pipeline.log
echo.
echo   To open the app: http://localhost:8000
echo   Analytics hub:   http://localhost:8501
echo.
echo   To remove auto-start, run: uninstall_autostart.bat
echo  ================================================================
echo.
pause
endlocal

@echo off
echo Removing DeepRetail auto-start tasks...
schtasks /Delete /TN "DeepRetail\Backend"  /F
schtasks /Delete /TN "DeepRetail\Pipeline" /F
if exist "%~dp0launch_backend.vbs"  del "%~dp0launch_backend.vbs"
if exist "%~dp0launch_pipeline.vbs" del "%~dp0launch_pipeline.vbs"
echo Done. DeepRetail will no longer auto-start on login.
pause

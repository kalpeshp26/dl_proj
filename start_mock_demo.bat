@echo off
title DeepRetail Mock Demo Launcher
echo ========================================================
echo       DeepRetail - Autonomous Checkout System
echo               MOCK DEMO MODE
echo ========================================================
echo.
echo Starting FastAPI Backend, Streamlit Dashboard, and Mock Pipeline...
echo No camera or model weights needed.
echo Dashboard URL: http://localhost:8501
echo Backend API  : http://127.0.0.1:8000/docs
echo.
python run_demo.py --mock --reset
pause

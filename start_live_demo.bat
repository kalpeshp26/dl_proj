@echo off
title DeepRetail Live Demo Launcher
echo ========================================================
echo       DeepRetail - Autonomous Checkout System
echo               LIVE WEBCAM MODE
echo ========================================================
echo.
echo Starting DeepRetail Enterprise System:
echo  - Enterprise Kiosk URL: http://localhost:8000
echo  - Streamlit Dashboard : http://localhost:8501
echo  - Backend API Docs    : http://127.0.0.1:8000/docs
echo  - Camera Index        : 0 (default webcam)
echo.
python run_demo.py --reset --anomaly --source 0 --no-display %*
pause

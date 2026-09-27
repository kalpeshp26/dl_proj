@echo off
title DeepRetail All-in-One Live Retail System
echo ========================================================
echo       DeepRetail - Autonomous Checkout System
echo             ALL-IN-ONE WEB DASHBOARD
echo ========================================================
echo.
echo Starting DeepRetail Enterprise System:
echo  - Enterprise Kiosk Web App: http://localhost:8000
echo  - Streamlit Analytics Hub : http://localhost:8501
echo  - Live Computer Vision     : Camera 0
echo.
echo Opening Enterprise Kiosk in your browser automatically...
echo Press Ctrl+C in this terminal to stop.
echo.
python run_demo.py --reset --anomaly --source 0 --no-display
pause

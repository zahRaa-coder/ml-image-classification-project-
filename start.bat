@echo off
cd /d "%~dp0"
echo Starting EcoScan AI ... (close this window to stop)
start "" cmd /c "timeout /t 8 >/dev/null & start http://127.0.0.1:5000"
python app/app.py
pause

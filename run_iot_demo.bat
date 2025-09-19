@echo off
echo Starting Motor IoT Monitoring Demo...
echo.

echo Installing required packages...
pip install flask pandas scikit-learn xgboost requests

echo.
echo Starting Flask app...
start "Flask App" cmd /k "python app.py"

echo.
echo Waiting 5 seconds for Flask to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting IoT Device Simulator...
start "IoT Simulator" cmd /k "python iot_simulator.py"

echo.
echo Demo started! 
echo - Flask app: http://127.0.0.1:5000
echo - IoT simulator is sending data every 2 seconds
echo - Go to Motor Monitor tab and enable "Live analysis"
echo.
echo Press any key to stop all processes...
pause > nul

echo Stopping processes...
taskkill /f /im python.exe
echo Demo stopped.







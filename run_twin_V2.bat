@echo off
echo ===================================================
echo      Starting Digital Twin System 
echo ===================================================

echo [1/3] Starting Flutter Dashboard (Takes time to compile)...
start cmd /k "cd _02_mobile_app && flutter run -d edge"

timeout /t 5 /nobreak > nul

echo [2/3] Starting AI Edge Engine...
start cmd /k "cd 01_AI_and_Data\src && conda activate digital_twin && python test_data_streamer.py"

echo.
echo ===================================================
echo [PAUSE] WAITING FOR DASHBOARD TO LOAD...
echo Please look at your browser. When the Flutter dashboard is fully open,
echo press any key here to start the Webots Simulation.
echo ===================================================
pause

echo [3/3] Starting Webots Physical Simulation...
start cmd /k "cd 04_3D_Simulation\worlds && motor_twin.wbt"

echo All systems launched! You can minimize this window.
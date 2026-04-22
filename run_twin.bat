@echo off
echo Starting Python Data Streamer...

:: Open a new terminal, activate the conda environment, and run the Python script
start cmd /k "cd 01_AI_and_Data\src && conda activate digital_twin && python test_data_streamer.py"

echo Starting Flutter Dashboard...

:: Open a second terminal and run the Flutter app on Microsoft Edge
start cmd /k "cd _02_mobile_app && flutter run -d edge"
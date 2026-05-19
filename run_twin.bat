@echo off
setlocal

:: ============================================================================
:: Digital Twin Launcher
:: 1. Creates a shortcut with the app icon in this folder (one-time setup).
:: 2. Starts the AI Inference Publisher.
:: 3. Waits 5 seconds for the publisher to connect and initialize.
:: 4. Opens the Webots 3D simulation.
:: ============================================================================

set "ROOT=%~dp0"
set "SHORTCUT=%ROOT%Run Digital Twin.lnk"

:: -- One-time shortcut creation with custom icon ------------------------------
if not exist "%SHORTCUT%" (
    echo Creating shortcut with custom icon...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Add-Type -AssemblyName System.Drawing; $b=[System.Drawing.Bitmap]::FromFile('%ROOT%_02_mobile_app\assets\app_icon.png'); $i=[System.Drawing.Icon]::FromHandle($b.GetHicon()); $f=[System.IO.File]::OpenWrite('%ROOT%_02_mobile_app\assets\app_icon.ico'); $i.Save($f); $f.Close(); $b.Dispose(); $ws=New-Object -ComObject WScript.Shell; $s=$ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath='%~f0'; $s.WorkingDirectory='%ROOT%'; $s.IconLocation='%ROOT%_02_mobile_app\assets\app_icon.ico'; $s.Save(); Write-Host 'Shortcut created.' } catch { Write-Host $_.Exception.Message }"
)

:: -- Step 1: Start the AI Inference Publisher in a new window -----------------
echo.
echo [1/2] Starting AI Inference Publisher...
start "AI Publisher" cmd /k "conda activate digital_twin && python _02_mobile_app\publisher_multi_machine.py"

:: -- Step 2: Wait 5 seconds for the publisher to initialize ------------------
echo [---] Waiting 5 seconds for the publisher to connect...
timeout /t 5 /nobreak >nul

:: -- Step 3: Open the Webots simulation ---------------------------------------
echo [2/2] Opening Webots simulation...
start "" "04_3D_Simulation\worlds\motor_twin.wbt"

echo.
echo Digital Twin launched. The publisher and Webots run independently.
echo You can close this window now.
pause

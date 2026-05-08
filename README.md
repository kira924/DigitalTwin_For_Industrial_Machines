# Digital Twin - Quick Start Guide

This guide provides the complete, simplified workflow to run the physics-aware Digital Twin project. This includes the Webots simulation, the MQTT broker, the Python backend, and the mobile application. **No Flutter SDK or Android Studio installation is required.**

## Prerequisites
1. **Webots:** Installed on your PC for the 3D simulation.
2. **Mosquitto MQTT Broker:** Installed on your host machine.
3. **Python Environment:** Set up with the necessary dependencies (e.g., `paho-mqtt`).
4. **Android Device:** To run the pre-built mobile application.

---

## Step 1: Configure Mosquitto Broker
Before running the broker, you must configure it to accept local network connections.
1. Install Mosquitto on your PC.
2. Navigate to the installation directory (usually in the `C:` drive, e.g., `C:\Program Files\mosquitto`).
3. Open the `mosquitto.conf` file with Administrator privileges.
4. Ensure the necessary modifications are made to allow external connections. Add or uncomment the following lines:
   ```text
   listener 1883
   allow_anonymous true

```

## Step 2: Install the Mobile Application

You only need to install the pre-compiled APK.

1. Locate the release APK in the project directory at this path:
`_02_mobile_app\build\app\outputs\flutter-apk\app-release.apk`
2. Transfer this `.apk` file to your Android phone.
3. Install the application.

## Step 3: Run the Backend Services

You will need to open **two separate command terminals** for this step.

**Terminal 1 (Run Mosquitto):**
Open a terminal, navigate to your Mosquitto folder, and start the broker using the configuration file:

```bash
mosquitto -v -c mosquitto.conf

```

**Terminal 2 (Run Python Publisher):**
Activate your Python environment (e.g., `digital_twin`), navigate to the app directory, and run the publisher script:

```bash
python .\publisher_multi_machine.py

```

## Step 4: Start the 3D Simulation

1. Open the Webots application.
2. Load the specific world file for this project located at:
`04_3D_Simulation\worlds\motor_twin.wbt`
3. Start the simulation in Webots.

## Step 5: View Live Telemetry

1. Ensure your Android phone and the host PC are connected to the **same Wi-Fi network**.
2. Open the Digital Twin app on your phone.
3. Navigate to the **CNC Machine** page. You should now see the live data streaming directly from the Webots simulation.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Industrial IoT Digital Twin with four integrated components:

1. **`01_AI_and_Data/`** — Python ML pipeline: LSTM model trained on NASA C-MAPSS dataset for Remaining Useful Life (RUL) prediction, with SHAP explainability and TFLite export.
2. **`_02_mobile_app/`** — Flutter mobile app (Android/iOS) with real-time MQTT telemetry, Firebase Auth/Firestore, and an embedded FastAPI backend.
3. **`03_Networking_MQTT/`** — Mosquitto broker config and test scripts.
4. **`04_3D_Simulation/`** — Webots 3D simulation controller that streams sensor data and reacts to AI diagnostics.

## Commands

### Python / AI

```bash
# Install all Python dependencies (from repo root)
pip install -r requirements.txt

# Run the AI inference publisher (bridges raw sensors → enriched telemetry)
python _02_mobile_app/publisher_multi_machine.py

# Start the FastAPI backend (image upload + on-demand inference)
uvicorn _02_mobile_app.backend.main:app --host 0.0.0.0 --port 8000

# Start the MQTT broker
mosquitto -v -c 03_Networking_MQTT/local_broker_setup/mosquitto.conf
```

### Flutter App

```bash
cd _02_mobile_app
flutter pub get
flutter run                          # run on connected device/emulator
flutter test                         # run all tests
flutter test test/widget_test.dart   # run a single test file
flutter build apk --release          # build Android APK
```

**Override API/MQTT config at build time:**
```bash
flutter run --dart-define=MQTT_BROKER=192.168.1.x --dart-define=API_BASE_URL=http://192.168.1.x:8000
```

## System Data Flow

```
Webots simulation (motor_twin.py)
  └─► MQTT: digital_twin/raw_sensors  (raw C-MAPSS sensor rows)
        └─► publisher_multi_machine.py  (LSTM inference + SHAP)
              └─► MQTT: digital_twin/engine_telemetry  (EngineTelemetryPayload JSON)
                    ├─► Flutter MqttService → AppState.ingestTelemetry()
                    └─► Webots: updates 3D model color from healthScore
```

The FastAPI backend (`_02_mobile_app/backend/main.py`) provides an alternative `/predict` HTTP endpoint for on-demand inference; it is separate from the MQTT pipeline.

## Flutter App Architecture

State management is a **single `AppState` ChangeNotifier** (`lib/app/state.dart`) exposed via `Provider`. All screens read from it with `context.watch<AppState>()`.

Key state relationships:
- **Firebase optional**: `AppState._initialize()` tries Firebase; if unavailable, falls back to the local `SharedPreferences` cache and seed data. `firebaseEnabled` flag gates all Firestore calls.
- **MQTT telemetry**: `MqttService` (`lib/services/mqtt_service.dart`) exposes a `Stream<EngineTelemetryPayload>`; the dashboard screen subscribes and calls `AppState.ingestTelemetry()`.
- **Machine → engine mapping**: `machineIdForEngine(int engineId)` translates the 1-based engine integer from the NASA publisher into a `machineId` string. `EngineTelemetryPayload` also accepts an explicit `machineId` string (Phase 3.3 schema).
- **Role-based access**: `AppRole` (admin / engineer / viewer). All write methods in `AppState` call `_blockWrite(permissions.canX, ...)` as defense-in-depth even when UI buttons are already gated.
- **Machine status** is auto-derived by `MachineStatusResolver`: live telemetry → `running`; no telemetry + report → `maintenance`; no telemetry + no report → `fault`. A 15-second sweep timer detects transitions without new payloads.

### Firestore Collections
- `machines/{id}` — machine docs, with subcollections `reports/` and `history/`
- `users/{uid}` — user profiles (role, isActive, email)
- `factory_settings` — global thresholds (temperature, vibration, power)

### MQTT Schema (Phase 3.3 telemetry payload)
Fields published to `digital_twin/engine_telemetry`:
```json
{
  "machineId": "cnc-01",
  "engine_id": 34,
  "current_cycle": 120,
  "predicted_rul": 87.5,
  "healthScore": 70.0,
  "status": "running",
  "temperature": 642.3,
  "vibration": 0.02,
  "current_sensor_readings": {"s_2": ..., "s_4": ..., ...},
  "ai_root_causes": {"s_4": 0.32, ...},
  "timestamp": "2026-05-15T10:00:00Z"
}
```
`EngineTelemetryPayload.fromJson()` accepts several key aliases; see `lib/models/engine_telemetry_payload.dart` for the full mapping.

## AI / ML Details

- **Dataset**: NASA C-MAPSS `test_FD001.txt` (aircraft turbofan engines). Located at `01_AI_and_Data/data/raw/CMAPSSData/`.
- **Model**: two-layer LSTM → Dense(1, linear). Input shape: `(30, 16)` (30-cycle sliding window, 16 features). Trained model: `01_AI_and_Data/saved_models/calibrated_model.keras`. Scaler: `saved_models/scalers/calibrated_scaler.gz`.
- **16 feature columns**: `setting_1`, `setting_2`, `s_2`, `s_3`, `s_4`, `s_7`, `s_8`, `s_9`, `s_11`, `s_12`, `s_13`, `s_14`, `s_15`, `s_17`, `s_20`, `s_21`.
- **RUL cap**: `MAX_USEFUL_RUL = 125`. Health% = `(predicted_rul / 125) * 100`, clamped 0–100.
- **Status thresholds**: health > 70 → running; 30–70 → warning; < 30 → fault.
- **SHAP**: `shap_explainer.py` wraps `DeepExplainer`; `extract_fault_causes()` returns the top contributing features.

## Network Configuration

| Service | Default address |
|---|---|
| MQTT broker | `127.0.0.1:1883` |
| FastAPI backend | `0.0.0.0:8000` |
| Android emulator → host | `10.0.2.2:8000` |
| Physical device | host LAN IP (override with `--dart-define`) |

For the broker to accept external connections, `mosquitto.conf` must contain `listener 1883` and `allow_anonymous true`.

## Webots Simulation

World file: `04_3D_Simulation/worlds/motor_twin.wbt`
Controller: `04_3D_Simulation/controllers/motor_twin/motor_twin.py`

The controller reads engine unit #34 from the C-MAPSS test dataset and replays it cycle by cycle at 0.5 s intervals, publishing raw rows to `digital_twin/raw_sensors`. It subscribes to `digital_twin/engine_telemetry` and maps `predicted_rul` to a health value (0–1) to update the PBR colors of the 3D motor body and sensor nodes.

## Strict Coding Standards (CRITICAL)

1. **Language:** ALL comments inside the code MUST be written in English. Do not use Arabic or any other language for code documentation or comments.
2. **No Emojis:** NEVER use emojis anywhere inside the codebase, including strings, comments, commit messages, or UI text.
3. **Clarity & Naming:** Prioritize readable variable names over short ones.
4. **Flutter UI Structure:** Break down complex Flutter screens into smaller, reusable widgets. Do not place all UI code in a single file. Maintain a clean architecture approach.
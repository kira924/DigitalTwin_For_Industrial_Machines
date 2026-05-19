# Predictive Digital Twin for Industrial Machines

**Graduation Project — Industrial IoT and Applied Artificial Intelligence**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture and End-to-End Workflow](#2-system-architecture-and-end-to-end-workflow)
3. [AI and Machine Learning Pipeline](#3-ai-and-machine-learning-pipeline)
4. [Flutter Dashboard — Architecture and Features](#4-flutter-dashboard--architecture-and-features)
5. [Tech Stack and Tools](#5-tech-stack-and-tools)
6. [Project Structure](#6-project-structure)
7. [Setup and Execution Guide](#7-setup-and-execution-guide)
8. [Key Configuration Reference](#8-key-configuration-reference)
9. [Defense Preparation — Design Decisions and Q&A Cheat Sheet](#9-defense-preparation--design-decisions-and-qa-cheat-sheet)

---

## 1. Project Overview

### The Industrial Problem

Industrial machines degrade over time. In traditional maintenance strategies, engineers either wait for a machine to fail before acting (reactive maintenance) or replace parts on a fixed schedule regardless of actual condition (preventive maintenance). Both approaches are costly: reactive maintenance causes unplanned downtime and safety hazards; preventive maintenance wastes resources on components that still have significant useful life remaining.

### The Proposed Solution

This project proposes and implements a **Predictive Digital Twin** — a real-time, AI-augmented virtual replica of a physical industrial machine. The system continuously monitors live sensor telemetry from the physical asset, predicts the machine's **Remaining Useful Life (RUL)** using a deep learning model, and presents actionable health intelligence to operators and engineers through a cross-platform dashboard. When a machine's health deteriorates, the system not only raises an alert but also identifies the **specific sensor channels most responsible for the degradation** using explainable AI (SHAP values), enabling targeted and efficient maintenance interventions.

### Core Goals

| Goal | Implementation |
|---|---|
| Predict machine RUL before failure occurs | Two-layer LSTM trained on NASA C-MAPSS turbofan dataset |
| Explain which components are causing degradation | SHAP GradientExplainer integration |
| Visualize machine health in real time | Flutter web/mobile dashboard with live MQTT telemetry |
| Simulate a physical machine environment | Webots 3D simulation streaming real C-MAPSS sensor cycles |
| Support multi-role industrial operations | Role-based access (Admin, Engineer, Viewer) with Firebase Auth |
| Operate in both cloud and air-gapped environments | HiveMQ Cloud (TLS) or local Mosquitto broker, switchable at runtime |

### Dataset

The project uses the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset, specifically the **FD001 subset**. Although the dataset originates from aircraft turbofan engine research, its degradation mechanics — progressive wear under controlled operational conditions, multi-sensor measurement, and run-to-failure trajectories — are directly transferable to any rotating industrial machine. The dataset provides:

- **100 training engines** with full run-to-failure trajectories (20,631 total cycles)
- **100 test engines** with trajectories truncated at a random point before failure
- **26 columns per row:** unit ID, cycle number, 3 operational settings, 21 sensor measurements
- **Ground truth RUL file** (`RUL_FD001.txt`) for quantitative model evaluation
- **One operating condition** (sea level), **one fault mode** (HPC degradation)

> **Academic Reference:** Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO.

---

## 2. System Architecture and End-to-End Workflow

### High-Level Architecture

```
+---------------------------+     MQTT: digital_twin/raw_sensors      +-----------------------------+
|   Webots 3D Simulation    | ------->------>------>------>--------->  |  Python AI Publisher        |
|   motor_twin.py           |                                          |  publisher_multi_machine.py |
|                           |                                          |                             |
|  - Loads test_FD001.txt   |                                          |  - Receives raw cycle rows  |
|  - Replays engine unit 34 |                                          |  - Assembles 30-cycle window|
|  - Publishes raw sensors  |     MQTT: digital_twin/engine_telemetry  |  - MinMaxScaler normalization|
|  - Receives health score  | <-------<-------<-------<-------<------  |  - LSTM RUL inference       |
|  - Updates 3D model color |                                          |  - SHAP attribution         |
+---------------------------+                                          |  - Publishes enriched JSON  |
                                                                       +-----------------------------+
                                                                                    |
                                                                                    | MQTT: digital_twin/engine_telemetry
                                                                                    |
                                                                                    v
                                                                       +-----------------------------+
                                                                       |  Flutter Dashboard          |
                                                                       |  (_02_mobile_app)           |
                                                                       |                             |
                                                                       |  - MqttService subscribes   |
                                                                       |  - AppState.ingestTelemetry |
                                                                       |  - ChangeNotifier rebuild   |
                                                                       |  - Firebase Firestore sync  |
                                                                       |  - Role-based UI rendering  |
                                                                       +-----------------------------+
```

### Step-by-Step Data Flow

**Step 1 — Webots Simulation (Raw Sensor Publication)**

The Webots controller (`04_3D_Simulation/controllers/motor_twin/motor_twin.py`) acts as the IoT edge node of a physical machine.

- It loads the C-MAPSS test dataset (`test_FD001.txt`) and selects **engine unit 34** as the simulated machine.
- The first **29 cycles are skipped** (publication starts at cycle 30) to fast-forward past the healthy warm-up period and reach the operationally interesting degradation phase sooner.
- Each simulation tick, the controller reads one row from the dataset (one operational cycle) and publishes it as a JSON message to the MQTT topic `digital_twin/raw_sensors`.
- The publish rate follows a **predefined cycle-based schedule** that simulates real-world variations in reporting frequency:

| Cycle Range | Publish Interval | Simulated Condition |
|---|---|---|
| 30 – 59 | 0.5 seconds | Active monitoring phase |
| 60 – 79 | 1.0 second | Reduced rate |
| 80 – 132 | 0.5 seconds | Active monitoring resumed |
| 133 – 142 | 1.0 second | Reduced rate |
| 143 – 169 | 0.5 seconds | Critical monitoring phase |
| 170 to end | 1.0 second | Final degradation phase |

- The controller simultaneously **subscribes** to `digital_twin/engine_telemetry` to receive the AI-computed health score. It maps `predicted_rul` to a normalized health index (0.0–1.0) and applies it to the **Physically Based Rendering (PBR) materials** of the 3D motor body and its sensor nodes, transitioning smoothly from steel-grey (healthy) through amber (warning) to deep red (critical).

**Raw payload example (published to `digital_twin/raw_sensors`):**
```json
{
  "engine_id": 34,
  "cycle": 45,
  "settings": [0.0009, -0.0002, 100.0],
  "sensors": [518.67, 642.66, 1574.94, 1402.56, 14.62, 21.61,
              554.68, 2388.02, 9033.22, 1.30, 47.47, 521.66,
              2388.02, 8138.62, 8.4195, 0.03, 392.0, 2388.0,
              100.0, 39.06, 23.419]
}
```

**Step 2 — MQTT Broker (Message Routing)**

All messages are routed through the MQTT broker. The system supports two broker configurations:

- **Production (HiveMQ Cloud):** `1c2024b173114f9d9e1577e9d4a5c467.s1.eu.hivemq.cloud`, port `8883`, TLS-encrypted, username/password authenticated.
- **Local/Development (Mosquitto):** `127.0.0.1:1883` (TCP) and `127.0.0.1:9001` (WebSocket), anonymous access, no TLS.

The broker is a pure message router. It decouples all system components — the Webots simulation, the publisher, and the Flutter app do not need to know each other's IP addresses. Any component can join or leave independently.

**Step 3 — AI Inference Engine (Python Publisher)**

`publisher_multi_machine.py` is the core AI processing component, bridging raw sensor data with health intelligence.

1. **Feature Assembly:** Extracts 16 specific sensor channels from the raw payload, dropping the 7 low-variance, constant channels identified during EDA.
2. **Normalization:** Applies the pre-fitted `MinMaxScaler` (`calibrated_scaler.gz`, feature range `[-1, 1]`) to the 16 features.
3. **Sliding Window Buffer:** Appends the scaled vector to a 30-cycle FIFO buffer. Inference begins only once 30 cycles have accumulated.
4. **LSTM Inference:** Passes the `(1, 30, 16)` tensor to `calibrated_model.keras` to obtain `predicted_rul`.
5. **Health Computation:** `health = clamp((predicted_rul / 125.0) * 100, 0, 100)`.
6. **SHAP Attribution:** Runs `extract_fault_causes()` to identify the top contributing sensor channels.
7. **Publication:** Publishes the enriched payload to `digital_twin/engine_telemetry`, which both the Flutter app and the Webots simulation consume.

**Enriched payload (published to `digital_twin/engine_telemetry`):**
```json
{
  "machineId": "cnc-01",
  "engine_id": 34,
  "current_cycle": 75,
  "predicted_rul": 87.5,
  "healthScore": 70.0,
  "status": "running",
  "temperature": 642.66,
  "vibration": 1402.56,
  "timestamp": "2026-05-15T10:00:00Z",
  "current_sensor_readings": {
    "setting_1": 0.0009, "setting_2": -0.0002,
    "s_2": 642.66, "s_3": 1574.94, "s_4": 1402.56,
    "s_7": 554.68, "s_8": 2388.02, "s_9": 9033.22
  },
  "ai_root_causes": {
    "s_3": 32.4, "s_14": 28.1, "s_4": 19.7
  }
}
```

**Step 4 — Flutter Dashboard (Real-Time UI Rendering)**

On each incoming `engine_telemetry` message:

1. `MqttService` delivers the raw JSON string to `AppState.ingestTelemetry()`.
2. The payload is deserialized into `EngineTelemetryPayload` and cached in `_latestPayloadByMachine`.
3. The current timestamp is recorded in `_lastSeenByMachine`, which drives the `MachineStatusResolver`.
4. `notifyListeners()` triggers a full reactive rebuild of all `context.watch<AppState>()` consumers.
5. The machine's status badge, health KPI, SHAP chart, sensor tiles, and live-dot freshness indicator all update simultaneously without any manual refresh logic.

---

## 3. AI and Machine Learning Pipeline

### 3.1 Dataset Preparation and Feature Engineering

**Source:** `01_AI_and_Data/src/data_preprocessing.py`

**RUL Label Generation:**
```
RUL(engine, cycle) = max_cycle_for_that_engine - current_cycle
```

**RUL Capping at 125 Cycles:**
Early in a machine's life, the RUL could be 300 or 400 cycles. These large values are replaced with the cap value of 125. This implements the assumption that the model only needs to be accurate within the actionable maintenance horizon. Cycles with RUL > 125 are assigned RUL = 125. This technique (piecewise linear target transformation) significantly improves model convergence and practical accuracy near the failure boundary.

**Low-Variance Sensor Removal:**
Seven channels are constant or near-constant under FD001's single operating condition and contribute zero predictive signal:

| Removed Channel | Physical Meaning | Observed Range |
|---|---|---|
| `s_1` (T2) | Total temperature at fan inlet | Constant: 518.67 |
| `s_5` (P2) | Pressure at fan inlet | Constant: 14.62 |
| `s_6` (P15) | Total pressure in bypass duct | 21.60 – 21.61 |
| `s_10` (epr) | Engine pressure ratio | Constant: 1.30 |
| `s_16` (farB) | Burner fuel-air ratio | Constant: 0.03 |
| `s_18` (Nf\_dmd) | Demanded fan speed | Constant: 2388.0 |
| `s_19` (PCNfR\_dmd) | Demanded corrected fan speed | Constant: 100.0 |
| `setting_3` | Throttle resolver angle | Constant: 100.0 |

**The 16 Retained Feature Columns (in order):**
`setting_1`, `setting_2`, `s_2`, `s_3`, `s_4`, `s_7`, `s_8`, `s_9`, `s_11`, `s_12`, `s_13`, `s_14`, `s_15`, `s_17`, `s_20`, `s_21`

**MinMaxScaler Normalization:**
A `MinMaxScaler` with `feature_range=(-1, 1)` is fitted on the full training set and serialized to `calibrated_scaler.gz`. The exact fitted bounds (from the scaler's `data_min_` and `data_max_` attributes) are:

| Sensor Key | Symbol | Unit | data\_min | data\_max |
|---|---|---|---|---|
| s\_2 | T24 | degrees R | 641.27 | 644.53 |
| s\_3 | T30 | degrees R | 1574.80 | 1612.11 |
| s\_4 | T50 | degrees R | 1387.16 | 1438.51 |
| s\_7 | P30 | psia | 550.70 | 555.57 |
| s\_8 | Nf | rpm | 2387.92 | 2388.32 |
| s\_9 | Nc | rpm | 9033.22 | 9203.22 |
| s\_11 | Ps30 | psia | 46.93 | 48.38 |
| s\_12 | phi | pps/psi | 519.48 | 523.26 |
| s\_13 | NRf | rpm | 2387.92 | 2388.35 |
| s\_14 | NRc | rpm | 8110.93 | 8259.42 |
| s\_15 | BPR | dimensionless | 8.3428 | 8.5462 |
| s\_17 | htBleed | dimensionless | 389.00 | 399.00 |
| s\_20 | W31 | lbm/s | 38.23 | 39.29 |
| s\_21 | W32 | lbm/s | 22.9562 | 23.6005 |

**Note on the narrow physical ranges:** FD001 has a single operating condition (sea level). This is why sensors like Nf (fan speed) span only 0.40 rpm (2387.92–2388.32). The tight clustering reflects highly controlled test conditions, not measurement error.

### 3.2 LSTM Model Architecture

**Source:** `01_AI_and_Data/src/train_base_model.py`

```
Input:   (30, 16)    -- 30 sequential time steps, 16 features per step

LSTM Layer 1:  128 units, return_sequences=True   (passes sequence to next layer)
Dropout:       0.3

LSTM Layer 2:  64 units, return_sequences=False   (outputs final hidden state)
Dropout:       0.3

Dense Output:  1 unit, linear activation          -- regression: predicts RUL value

Loss Function:  Mean Squared Error (MSE)
Optimizer:      Adam (adaptive learning rate)
```

**Training Configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| Sequence Length (inference) | 30 cycles | Balances context with early-cycle availability |
| Sequence Length (base training) | 60 cycles | More historical context during initial training |
| Batch Size | 128 | Stable gradient estimates for time-series regression |
| Maximum Epochs | 50 | Upper bound; early stopping terminates before this |
| Early Stopping patience | 10 epochs | Prevents overfitting; monitors val\_loss |
| ReduceLROnPlateau factor | 0.5 | Halves LR when val\_loss plateaus |
| ReduceLROnPlateau patience | 5 epochs | More aggressive than early stopping |
| Dropout rate | 0.3 | Regularization for single-fault-mode dataset |
| Validation split | 10% | Hold-out subset of training trajectories |

**Saved Model Artifacts:**

| File | Purpose |
|---|---|
| `01_AI_and_Data/saved_models/calibrated_model.keras` | Production LSTM inference model |
| `01_AI_and_Data/saved_models/calibrated_model.tflite` | Mobile-optimized TFLite export |
| `01_AI_and_Data/saved_models/scalers/calibrated_scaler.gz` | Fitted MinMaxScaler |
| `_02_mobile_app/backend/calibrated_model.keras` | Copy used by FastAPI backend |

### 3.3 Transfer Learning for Multi-Dataset Generalization

**Source:** `01_AI_and_Data/src/transfer_learning.py`

To extend the model beyond the single FD001 condition, a transfer learning approach was developed:

1. Load the pre-trained FD001 model.
2. Freeze the first LSTM layer (preserves general temporal feature detectors).
3. Unfreeze all subsequent layers for domain-specific fine-tuning.
4. Fit a new `MinMaxScaler` on FD002 data (FD002 has 6 operating conditions with wider sensor ranges).
5. Fine-tune for 40 epochs on 20 engines from FD002 with learning rate 0.001.
6. Evaluate against FD002 ground truth RUL.

This demonstrates that the architecture generalizes across machine operating regimes with minimal additional labeled data.

### 3.4 TFLite Export for Edge and Mobile Deployment

**Source:** `01_AI_and_Data/src/export_tflite.py`

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS   # required for LSTM support
]
```

Post-training quantization reduces model size significantly while maintaining acceptable accuracy, enabling on-device inference without a network connection.

### 3.5 SHAP Explainability — Root Cause Analysis

**Source:** `01_AI_and_Data/src/shap_explainer.py`

SHAP (SHapley Additive exPlanations) provides mathematically rigorous feature attribution based on cooperative game theory. It answers: "How much did each sensor contribute to this specific RUL prediction?"

**Initialization:**
```python
# background_data shape: (10, 30, 16) -- 10 representative baseline windows
explainer = shap.GradientExplainer(model, background_data)
```

**Inference:**
```python
# input_window shape: (1, 30, 16)
shap_values = explainer.shap_values(input_window)
# Result shape: (1, 30, 16)

# Aggregate across the 30 time steps
mean_abs_shap = np.abs(shap_values[0]).mean(axis=0)  # shape: (16,)

# Normalize to percentages and return top contributors
```

**Output in telemetry payload:**
```json
"ai_root_causes": {
  "s_3": 32.4,
  "s_14": 28.1,
  "s_4": 19.7
}
```

The Flutter dashboard renders these as the **"Key Risk Factors"** bar chart, using the engineering symbol (T30, NRc, T50) rather than the raw JSON key. The chart shows the top 4 contributing sensors.

### 3.6 FastAPI HTTP Backend (Alternative Inference Path)

**Source:** `_02_mobile_app/backend/main.py`

An HTTP endpoint for stateless, on-demand inference — independent of the MQTT pipeline:

```
POST /predict
{
  "machine_id": "cnc-01",
  "engine_id": 34,
  "current_cycle": 75,
  "series_data": [ [16 values] x 30 rows ]
}
```

This path is used for image-upload workflows and on-demand analysis when a full MQTT stream is not available.

---

## 4. Flutter Dashboard — Architecture and Features

### 4.1 State Management

The entire application state is a **single `AppState` ChangeNotifier** (`lib/app/state.dart`), exposed globally via the `Provider` package. It is split into logical extension files using Dart's `part` system:

| Part File | Responsibility |
|---|---|
| `state_firebase.dart` | Firebase/Firestore initialization, real-time subscriptions |
| `state_auth.dart` | Login, logout, role resolution, session persistence |
| `state_telemetry.dart` | MQTT payload ingestion, per-machine telemetry cache, SHAP handling |
| `state_machines.dart` | Fleet-level metrics (running/fault/maintenance counts, OEE) |
| `state_users.dart` | User management, role assignment |
| `state_reports.dart` | Maintenance report CRUD |
| `state_ui.dart` | Navigation tab state, theme mode, dashboard time range |

### 4.2 Machine Status Resolution

Machine status is **never set manually**. `MachineStatusResolver` derives it from two observable facts:

```
live telemetry (timestamp within 30 seconds)          -->  running
no live telemetry  AND  machine has an open report     -->  maintenance
no live telemetry  AND  no report                      -->  fault
```

A **15-second sweep timer** re-evaluates all machine statuses periodically. This means a machine transitions automatically from `running` to `fault` even without any new MQTT messages — a critical safety property for real industrial deployments.

All consumers of status (machine card badges, filter chips, count pills) call `resolvedStatusFor(machineId)` so there is exactly one source of truth and no stale display values.

### 4.3 Role-Based Access Control

| Role | Capabilities |
|---|---|
| **Admin** | Manage users, machines, global thresholds; full read access |
| **Engineer** | Write reports, act on machines, read all telemetry and alerts |
| **Viewer** | Read-only access to all screens; no write actions |

All write methods in `AppState` call an internal `_blockWrite()` guard regardless of UI-level visibility, providing defense-in-depth.

### 4.4 Firebase Integration (Optional)

Firebase is treated as **optional resilient infrastructure**. The app boots from local seed data and a `SharedPreferences` cache. Firebase connection is attempted asynchronously. If unavailable, the app continues in offline mode with no user-visible error.

**Firestore Collections:**

| Collection | Purpose |
|---|---|
| `machines/{id}` | Machine documents |
| `machines/{id}/reports/` | Per-machine maintenance reports |
| `machines/{id}/history/` | Status change history |
| `users/{uid}` | User profiles (role, isActive, email) |
| `factory_settings` | Global thresholds (temperature, vibration, power) |

### 4.5 Dashboard Screens

| Screen | Key Features |
|---|---|
| **Dashboard** | Fleet KPIs, at-risk machine table, health trends, sensor feed |
| **Machines** | Responsive grid, live status badges, freshness live-dot, 4-way status filter, text search |
| **Machine Detail** | Health / Cycle / Predictive Maintenance KPIs, Key Risk Factors chart (SHAP Top 4), Live Telemetry 4-tile grid, All Sensors sheet (14 sensors, reactive), History timeline, Reports, Admin actions |
| **Alerts** | Emergency alert feed with severity color coding and deduplication |
| **History** | Fleet-wide status transition timeline |
| **Admin / Settings** | Profile management, app behavior, system status, machine CRUD, user management |

### 4.6 Live Telemetry Display Details

**4-Tile Summary Grid (always visible, layout never changes regardless of data state):**

| Tile | Live Source (payload present) | Fallback (awaiting data) | Unit |
|---|---|---|---|
| Temp | `payload.temperature` or `s_2` | `machine.temperature` | degrees R |
| Speed | `payload.speed` or `s_8` | `machine.speed` | rpm |
| Pressure | `s_7` or `s_11` | `machine.pressure` | psia |
| Efficiency | `payload.healthScore` | `machine.efficiency` | % |

**All Sensors Sheet (14 C-MAPSS channels):**
- Opened via the "More sensors" button.
- Uses `Consumer<AppState>` inside a `ChangeNotifierProvider.value`-wrapped modal bottom sheet, so sensor values update in real time as new MQTT packets arrive without closing the sheet.
- Values are displayed directly in physical engineering units (no transformation). The publisher emits raw physical values; the Flutter app displays them as received.

---

## 5. Tech Stack and Tools

### Languages

| Language | Version | Used For |
|---|---|---|
| Python | 3.10 (Anaconda) | AI/ML pipeline, Webots controller, MQTT publisher, FastAPI |
| Dart | >= 3.3.0 | Flutter application (all screens, state, services) |
| VRML / Webots DSL | R2025a | 3D simulation world definition (.wbt file) |

### Python Libraries — AI and Data Science

| Library | Version | Purpose |
|---|---|---|
| TensorFlow / Keras | 2.15.0 | LSTM model definition, training, and inference |
| scikit-learn | 1.8.0 | MinMaxScaler, preprocessing |
| SHAP | 0.49.1 | GradientExplainer for feature attribution |
| NumPy | 2.4.4 | Tensor and array operations |
| Pandas | 3.0.2 | Dataset loading and feature engineering |
| joblib | 1.5.3 | Model and scaler serialization / deserialization |
| Matplotlib | 3.10.9 | Training curves, EDA visualizations |

### Python Libraries — Networking and Backend

| Library | Version | Purpose |
|---|---|---|
| paho-mqtt | 2.1.0 | MQTT client (CallbackAPIVersion.VERSION2) |
| FastAPI | 0.136.1 | HTTP inference REST endpoint |
| uvicorn | 0.46.0 | ASGI server for FastAPI |
| firebase-admin | 7.4.0 | Server-side Firestore access |
| Cloudinary | 1.44.2 | Machine and user image cloud storage |
| python-dotenv | 1.2.2 | Environment variable management |

### Flutter / Dart Packages

| Package | Version | Purpose |
|---|---|---|
| provider | ^6.1.2 | ChangeNotifier + Provider state management |
| mqtt\_client | ^10.0.0 | MQTT WebSocket and TCP client |
| fl\_chart | ^0.66.2 | Bar charts (SHAP), line charts (health trends) |
| firebase\_core | ^3.15.2 | Firebase SDK initialization |
| firebase\_auth | ^5.6.2 | Email/password authentication |
| cloud\_firestore | ^5.6.11 | Real-time Firestore database |
| shared\_preferences | ^2.5.3 | Local persistence and offline cache |
| http | ^1.2.0 | FastAPI backend HTTP calls |
| image\_picker | ^1.1.2 | Profile photo and machine image upload |
| flutter\_launcher\_icons | ^0.13.1 | App icon generation for Android and iOS |

### Infrastructure and Protocols

| Component | Technology | Configuration |
|---|---|---|
| MQTT Broker (Cloud) | HiveMQ Cloud | Port 8883, TLS, authenticated |
| MQTT Broker (Local) | Eclipse Mosquitto | Port 1883 TCP + 9001 WebSocket |
| IoT Protocol | MQTT v5 (paho 2.x) | QoS 1, retain=false for telemetry |
| 3D Simulation Engine | Webots R2025a | Physics-based robot simulation platform |
| Backend API | FastAPI + uvicorn | REST, JSON, CORS enabled for all origins |
| Auth and Database | Firebase (Google Cloud) | Firestore + Firebase Auth |
| Image Storage | Cloudinary | Machine profile images, user avatars |

---

## 6. Project Structure

```
Digital_Twin/
|
+-- 01_AI_and_Data/                         AI and ML module
|   +-- src/
|   |   +-- data_preprocessing.py           Dataset loading, RUL computation, feature selection
|   |   +-- train_base_model.py             LSTM architecture, training loop, callbacks
|   |   +-- transfer_learning.py            Fine-tuning for FD002 generalization
|   |   +-- shap_explainer.py               SHAP GradientExplainer setup and extraction
|   |   +-- export_tflite.py                Keras to TFLite conversion with quantization
|   |   +-- test_data_streamer.py           Standalone local AI edge node (alternative publisher)
|   +-- saved_models/
|   |   +-- calibrated_model.keras          Production LSTM model
|   |   +-- calibrated_model.tflite         Mobile-optimized TFLite export
|   |   +-- scalers/calibrated_scaler.gz    Fitted MinMaxScaler (feature_range=(-1,1))
|   +-- data/raw/CMAPSSData/
|   |   +-- train_FD001.txt                 100 engines, 20,631 cycles
|   |   +-- test_FD001.txt                  100 engines, truncated trajectories
|   |   +-- RUL_FD001.txt                   Ground truth RUL values
|   |   +-- Damage Propagation Modeling.pdf PHM08 source paper
|   +-- notebooks/
|       +-- 01_EDA_and_Cleaning.ipynb
|       +-- 02_LSTM_Training.ipynb
|
+-- _02_mobile_app/                         Flutter app + Python publisher
|   +-- publisher_multi_machine.py          AI inference publisher (main MQTT bridge)
|   +-- lib/
|   |   +-- app/                            AppState + all state extension part files
|   |   +-- models/                         Data models (Machine, Payload, User, Report...)
|   |   +-- data/                           Sensor catalog, seed data, descaler utility
|   |   +-- services/                       MQTT, Firebase sync, status resolver
|   |   +-- screens/                        All UI screens
|   |   +-- theme/                          Design tokens, color palette, shared widgets
|   |   +-- widgets/                        Reusable components
|   +-- backend/
|   |   +-- main.py                         FastAPI inference server
|   |   +-- calibrated_model.keras          Backend model copy
|   |   +-- calibrated_scaler              Backend scaler copy
|   +-- assets/
|   |   +-- images/                         Machine photos for seed data
|   |   +-- app_icon.png                    Application icon (source PNG)
|   |   +-- app_icon.ico                    Generated ICO for Windows shortcut
|   +-- pubspec.yaml                        Flutter dependency manifest
|
+-- 03_Networking_MQTT/                     Broker configuration
|   +-- local_broker_setup/mosquitto.conf   Port 1883 (TCP) + 9001 (WebSocket)
|   +-- scripts/                            Standalone test publisher and subscriber
|
+-- 04_3D_Simulation/                       Webots simulation
|   +-- controllers/motor_twin/
|   |   +-- motor_twin.py                   Webots IoT edge node controller
|   +-- worlds/motor_twin.wbt               Webots 3D scene definition
|   +-- workspace/train_FD001.txt           Dataset copy for Webots workspace
|
+-- 05_Docs_and_Specs/integration_notes/    Technical documentation
|   +-- sensor_map.json                     C-MAPSS sensor catalog with physical ranges
|   +-- sensors_details.txt                 Sensor table (symbol, unit, range)
|   +-- cmapss_descaler.py                  Python inverse-scaling utility
|   +-- inverse_min_max_scaling.txt         Scaling formula reference
|   +-- App_Edits.txt                       UI feature requirements checklist
|
+-- requirements.txt                        Root Python dependencies
+-- run_twin.bat                            System launcher script
+-- Run Digital Twin.lnk                   Windows shortcut with custom app icon
+-- CLAUDE.md                              Architecture guide
```

---

## 7. Setup and Execution Guide

### Prerequisites

| Requirement | Version / Notes |
|---|---|
| Anaconda or Miniconda | For the `digital_twin` Python environment |
| Python | 3.10 (managed by conda) |
| Flutter SDK | >= 3.3.0 |
| Webots | R2025a (registers `.wbt` file association on install) |
| Eclipse Mosquitto | Only if using local broker (Option B below) |

### Step 1 — Python Environment

```bash
conda create -n digital_twin python=3.10
conda activate digital_twin
pip install -r requirements.txt
```

### Step 2 — MQTT Broker

**Option A — HiveMQ Cloud (default, zero configuration):**
The publisher and Webots controller are preconfigured. No action needed.

**Option B — Local Mosquitto (air-gapped / offline):**
```bash
# Start the broker
mosquitto -v -c 03_Networking_MQTT/local_broker_setup/mosquitto.conf

# Set environment variables before running publisher and Webots
set MQTT_BROKER=127.0.0.1
set MQTT_PORT=1883
set MQTT_USE_TLS=false
set MQTT_USERNAME=
set MQTT_PASSWORD=
```

### Step 3 — Start the AI Inference Publisher

```bash
conda activate digital_twin
cd D:\study\Uni_Matrial\Final_Project\Digital_Twin
python _02_mobile_app/publisher_multi_machine.py
```

Wait for the confirmation output:
```
AI Assets Loaded Successfully.
[MQTT] Connected successfully to ...
[MQTT] Subscribed to 'digital_twin/raw_sensors' -- waiting for Webots sensor data...
```

### Step 4 — Launch the Webots Simulation

```bash
webots 04_3D_Simulation/worlds/motor_twin.wbt
```

Or open Webots and load `04_3D_Simulation/worlds/motor_twin.wbt` via File > Open World.

The console will show:
```
[MQTT] Connected to ...hivemq.cloud:8883
[PUBLISH] Sent Cycle 30 | Interval: 0.5s | Health: 1.00
[PUBLISH] Sent Cycle 31 | Interval: 0.5s | Health: 1.00
```

After 30 cycles, the publisher responds:
```
Published Cycle 59 | Health: 94.3%
[INFO] Interval changed to 1.0s at cycle 60
```

### Step 5 — Run the Flutter Dashboard

```bash
cd _02_mobile_app
flutter pub get
flutter run -d chrome        # web browser
flutter run                  # connected Android/iOS device or emulator
flutter build apk --release  # build Android APK
```

**Override broker/API for physical device on a local network:**
```bash
flutter run --dart-define=MQTT_BROKER=192.168.1.x --dart-define=API_BASE_URL=http://192.168.1.x:8000
```

### One-Click Launch (Windows Only)

Double-click **`Run Digital Twin.lnk`** in the project root. This shortcut:
1. Activates the `digital_twin` conda environment.
2. Opens a terminal window running `publisher_multi_machine.py`.
3. Waits 5 seconds for the publisher to connect and load AI assets.
4. Launches Webots with `motor_twin.wbt` automatically.

### Optional — FastAPI Inference Backend

```bash
conda activate digital_twin
cd _02_mobile_app
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation: `http://localhost:8000/docs`

---

## 8. Key Configuration Reference

### Critical Numerical Constants

| Constant | Value | Location | Meaning |
|---|---|---|---|
| `MAX_USEFUL_RUL` | 125 cycles | publisher, backend, preprocessing | RUL cap; health% = (RUL / 125) x 100 |
| `SEQ_LENGTH` | 30 cycles | publisher, backend | LSTM sliding window size at inference |
| `FEATURES_COUNT` | 16 | publisher, backend | Sensor channels after feature selection |
| `kLiveTelemetryWindow` | 30 seconds | MachineStatusResolver | Telemetry within this window = machine is running |
| `_kCriticalRulCycles` | 30.0 | AppState | RUL below this triggers emergency alert |
| `_presentationRunningCount` | 4 | AppState | Machines held "running" by heartbeat for demo |
| Health thresholds | >70 running, 30-70 warning, <30 fault | publisher, backend, AppState | Identical across all three components |
| `SKIP_CYCLES` | 29 | motor\_twin.py | Skip first 29 rows; publish from cycle 30 |
| `TARGET_ROTOR_RPM` | 1000 | motor\_twin.py | Webots 3D rotor speed |
| `SELECTED_UNIT` | 34 | motor\_twin.py | Engine unit from test dataset |

### MQTT Topics

| Topic | Direction | Publisher | Subscriber |
|---|---|---|---|
| `digital_twin/raw_sensors` | Edge to Cloud | Webots controller | Python publisher |
| `digital_twin/engine_telemetry` | Cloud to Edge + App | Python publisher | Flutter app and Webots controller |

### Network Addresses

| Service | Default | Override Method |
|---|---|---|
| MQTT Broker (Cloud) | hivemq.cloud:8883 (TLS) | `MQTT_BROKER`, `MQTT_PORT`, `MQTT_USE_TLS` env vars |
| MQTT Broker (Local) | 127.0.0.1:1883 | Same env vars as above |
| MQTT WebSocket | 127.0.0.1:9001 | Mosquitto config |
| FastAPI Backend | 0.0.0.0:8000 | `API_BASE_URL` flutter dart-define |
| Android Emulator to Host | 10.0.2.2:8000 | Hardcoded fallback in Flutter config |

---

## 9. Defense Preparation — Design Decisions and Q&A Cheat Sheet

This section is your primary reference for answering architectural "why" questions from the examination committee.

---

### Why MQTT Instead of HTTP/REST for Sensor Data Streaming?

**The core reason is the publish-subscribe model combined with minimal protocol overhead.**

| Dimension | HTTP/REST | MQTT |
|---|---|---|
| Communication model | Request-response (client must poll) | Publish-subscribe (broker pushes to all subscribers) |
| Protocol overhead | 200-800 bytes per message (headers) | 2 bytes fixed header; total ~50-100 bytes |
| Multiple consumers | Requires separate request from each consumer | One publish reaches all subscribers simultaneously at zero extra cost |
| Connection handling | Stateless; new TCP connection per request | Persistent connection; minimal reconnect overhead |
| Industrial standard | Web APIs, CRUD applications | IoT, SCADA, industrial telemetry (IEC 20922) |

In this system, a single telemetry message from Webots needs to reach **both** the Python publisher AND the Flutter dashboard simultaneously. MQTT's topic model delivers this with one publish operation. With HTTP, you would need two separate POST requests or a complex event bus.

**QoS levels in MQTT** provide delivery guarantees without application-level acknowledgement logic: QoS 1 (used here) guarantees at-least-once delivery, which is appropriate for sensor telemetry where a duplicate is acceptable but a dropped message is not.

---

### Why LSTM Instead of Random Forest, XGBoost, or SVR?

**The fundamental reason is that RUL prediction is a sequential problem — the current state is meaningless without the trajectory that led to it.**

A current T30 reading of 1590°R means very different things depending on whether it was 1571°R ten cycles ago (stable, healthy) or 1610°R ten cycles ago (rapidly declining toward failure). Tabular models like Random Forest treat each row as independent. They have no mechanism to learn temporal dependencies.

| Model | Temporal Awareness | Handles Variable-Length History | Suitable for C-MAPSS |
|---|---|---|---|
| Random Forest | None (tabular) | No | Poor |
| XGBoost | None (tabular) | No (unless manual lag features) | Moderate with engineering |
| SVR | None | No | Moderate |
| 1D-CNN | Local patterns only | Fixed window | Good |
| **LSTM** | Long-range dependencies via gating | Fixed window (by design) | **Excellent** |
| Transformer | Long-range via attention | Fixed window | Good but data-hungry |

**Why not a Transformer?** Transformers require more data and compute to outperform LSTMs. With 100 training engines (FD001), an LSTM provides a better accuracy-to-complexity ratio and avoids overfitting.

**Why stacked (2-layer) LSTM?** The first layer learns low-level temporal patterns (short-duration sensor fluctuations). The second layer learns high-level temporal abstractions (multi-cycle degradation trends). This hierarchy is standard in deep sequential modeling.

---

### Why Integrate SHAP for Explainability?

**SHAP transforms the system from a "black-box predictor" into an "actionable diagnostic tool" — this is the difference between a research prototype and a deployable industrial system.**

A raw RUL prediction tells an engineer: "This machine will fail in 45 cycles." This is insufficient for action. SHAP answers: "The primary contributors to this prediction are T30 (HPC outlet temperature, 32.4%) and NRc (corrected core speed, 28.1%) — inspect the high-pressure compressor."

**Why SHAP specifically and not simpler attribution methods (e.g., gradient-based saliency)?**
- SHAP is grounded in **cooperative game theory (Shapley values)**. It satisfies four mathematical axioms: efficiency, symmetry, dummy property, and additivity. Simpler methods satisfy none.
- SHAP values are **additive**: the sum of all feature SHAP values equals the difference between the model's prediction and the expected prediction. This makes them interpretable as absolute contributions.
- **GradientExplainer** is used (not KernelExplainer) because it is computationally efficient for neural networks, using the model's gradient information rather than sampling the feature space.

---

### How Does the Webots Simulation Accurately Reflect Real-World Industrial Physics?

**The simulation is a replay of a validated physics-based simulation, not synthetic or randomized data.**

The C-MAPSS dataset was generated by NASA using the **Commercial Modular Aero-Propulsion System Simulation** — a high-fidelity thermodynamic model of a two-spool turbofan engine with a closed-loop controller, validated against real engine test data. Every sensor value replayed by Webots is the output of this validated simulation.

The degradation trajectory for engine unit 34 follows the exponential health degradation model described in the PHM08 paper (Equation 4): `h(t) = 1 - exp{a·t^b} - d`, where the parameters were randomly drawn from constrained distributions to ensure physical plausibility.

The **PBR color feedback** provides real-time visual confirmation that the AI health assessment is consistent with the physical state the simulation is producing — a closed-loop verification path.

---

### What is the RUL Cap of 125 Cycles and Why Was It Chosen?

**The cap implements a practical maintenance decision horizon and eliminates a training inefficiency.**

When a machine has 350 cycles of remaining life, the precise value (350 vs. 300) is irrelevant — the machine simply does not need attention. Without capping, the model wastes capacity trying to distinguish between large, low-priority RUL values where no action would be taken.

By capping at 125 cycles, the model concentrates its learning capacity on the degradation phase that matters: the final 125 cycles before failure, when maintenance decisions are actually being made.

**Why 125 specifically?** Analysis of the FD001 training trajectories shows that sensor values begin showing consistent, monotonic degradation trends approximately 125 cycles before failure. This is the inflection point beyond which early detection becomes feasible and actionable. This choice is consistent with established C-MAPSS benchmarking literature.

---

### Why is Machine Status Fully Automatic and Never Manually Set?

**Manual status fields become stale and create dangerous false confidence.**

If an operator forgets to update a machine status, or the interface is unavailable, the dashboard can display a machine as "Running" long after it has stopped transmitting data — which is precisely when an engineer most needs to know there is a problem.

`MachineStatusResolver` eliminates this entire class of error:
- Status is a pure function of two observable inputs: the last telemetry timestamp and the presence of an open maintenance report.
- It is recomputed every 15 seconds by a sweep timer, independently of any MQTT traffic.
- Every component that displays status (card badges, filter chips, count pills) calls the same resolver, guaranteeing consistency.

---

### Why Flutter for the Dashboard Instead of React, Angular, or a Native Platform?

**Flutter provides a single codebase with native performance across all target platforms.**

| Property | Flutter | React (Web) | Native Android/iOS |
|---|---|---|---|
| Target platforms | Android, iOS, Web, Desktop — one codebase | Web only | Two separate codebases |
| Rendering | Compiled to native ARM / CanvasKit (no WebView) | DOM-based | Native |
| Real-time UI rebuild | `ChangeNotifier` + `context.watch` — sub-millisecond | State management (Redux, etc.) — comparable | Comparable |
| Industrial deployability | Single APK for tablets, web for office machines | Browser-dependent | Requires two apps |

For an industrial deployment, the same Flutter app serves a supervisor's office desktop (web), a factory floor Android tablet, and an engineer's iOS phone without code duplication.

---

### Why Two Broker Options (HiveMQ Cloud vs. Local Mosquitto)?

**To support both connected environments and air-gapped industrial facilities.**

HiveMQ Cloud handles all infrastructure concerns (TLS termination, authentication, scaling, uptime) and is appropriate for cloud-connected smart factories.

Local Mosquitto is critical for environments where internet access is not permitted — common in high-security industrial sectors (automotive, defense, pharmaceuticals, utilities). The same application binary works in both environments by changing three environment variables (`MQTT_BROKER`, `MQTT_PORT`, `MQTT_USE_TLS`).

---

### Why Does the Publisher Send Physical Values Instead of Scaled Values?

**To decouple the display pipeline from the ML pipeline's internal preprocessing details.**

Sending scaled values (e.g., `-0.527` for T24) to the dashboard would force the Flutter app to know the scaler's parameters. If the scaler is retrained with different data or a different feature range, the dashboard would need to be updated simultaneously — tight coupling between two independent components.

By sending physical values (`642.7°R` for T24), the publisher provides a **stable, semantically meaningful contract**. The dashboard displays what it receives. The MinMaxScaler normalization exists only inside the Python inference engine, between receiving the raw payload and feeding the tensor to the LSTM. This is the correct application of the single-responsibility principle.

---

### What Would You Change With More Time?

Preparing this answer demonstrates engineering maturity and understanding of the system's boundaries.

- **Real hardware integration:** Replace the dataset replay with a live OPC-UA or Modbus connection to a physical PLC, reading actual vibration accelerometers, RTD temperature sensors, and pressure transducers.
- **Anomaly detection layer:** Add an unsupervised model (Autoencoder or Isolation Forest) as a first-stage filter to detect novel failure modes that the supervised LSTM was not trained on.
- **Multi-fault mode generalization:** Train a unified model on FD001–FD004 combined, or apply domain adaptation techniques for robust performance across operating conditions without per-machine fine-tuning.
- **Federated learning:** Allow each factory machine to contribute gradient updates to the global model without sharing raw sensor data, addressing privacy in multi-tenant industrial IoT.
- **Digital twin calibration loop:** Use Bayesian optimization to continuously tune the simulation's degradation parameters so the digital twin's state drift from the physical machine remains bounded over the machine's operational lifetime.
- **Streaming analytics:** Replace the single-machine pub/sub architecture with a proper stream processing framework (Apache Kafka + Flink) to scale to hundreds of machines simultaneously.

---

*End of README*
*Project: Predictive Digital Twin for Industrial Machines*
*Graduation Project — 2026*

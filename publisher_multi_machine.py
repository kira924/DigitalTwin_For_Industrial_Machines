"""
Phase 3.3 multi-machine MQTT publisher.

Publishes Phase 3.3 friendly-schema telemetry to one shared topic for
multiple machines. The Flutter app subscribes to the same topic and
routes each payload to the matching machine via `machineId`.

Topic
-----
    digital_twin/engine_telemetry  (HiveMQ public broker by default)

This matches the topic the Flutter `MqttService` subscribes to.

Payload (Phase 3.3 friendly schema)
-----------------------------------
    {
      "machineId": "M-001",
      "engine_id": 1,                 # legacy alias, also accepted
      "current_cycle": 142,
      "predicted_rul": 78.4,
      "temperature": 71.2,
      "vibration": 0.42,
      "power": 18.6,
      "speed": 1180,
      "status": "running",            # "running" | "fault" | "stopped"
      "timestamp": "2026-04-28T10:30:00Z",
      "current_sensor_readings": {    # legacy NASA C-MAPSS fields,
        "s_2": 71.2,                  # consumed by SHAP/AI cards
        "s_4": 0.42,
        "s_11": 18.6
      },
      "ai_root_causes": {             # SHAP-style attributions
        "s_2": 0.4, "s_4": 0.3, "s_11": 0.2
      }
    }

The machine pool is hard-coded to match the demo fleet's machine IDs
(see `lib/data/machine_catalog.dart`):
    cnc-01, robotic-02, hyd-03, pkg-04, lathe-05, wld-06,
    asm-07, conv-08

Each machine drifts independently so Flutter's per-machine cache shows
distinct telemetry, never one global value reused across cards.

Run
---
    pip install paho-mqtt
    python publisher_multi_machine.py

Optional env vars:
    MQTT_BROKER   default: broker.hivemq.com
    MQTT_PORT     default: 1883
    MQTT_TOPIC    default: digital_twin/engine_telemetry
    PUBLISH_HZ    default: 1.0   (publishes per second per machine)

NASA C-MAPSS dataset
--------------------
The original publisher (`publisher_engine1.py`) read from a 6.7 MB
`test_FD004.txt` file and only published engine 1. The dataset is no
longer required — this publisher synthesizes plausible drift in pure
Python so the demo runs without external assets. If you want real
C-MAPSS replay, copy the legacy publisher back and update the topic +
add `machineId`.
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

BROKER = os.environ.get("MQTT_BROKER", "broker.hivemq.com")
PORT = int(os.environ.get("MQTT_PORT", "1883"))
TOPIC = os.environ.get("MQTT_TOPIC", "digital_twin/engine_telemetry")
PUBLISH_HZ = float(os.environ.get("PUBLISH_HZ", "1.0"))

# Match the demo catalog machine IDs (lib/data/machine_catalog.dart).
# engine_id is the 1-based index used by the legacy NASA schema; it's
# kept for back-compat with the Flutter side's index lookup.
MACHINES = [
    {"machineId": "cnc-01",     "engine_id": 1, "name": "CNC Mill 01"},
    {"machineId": "robotic-02", "engine_id": 2, "name": "Robotic Arm 02"},
    {"machineId": "hyd-03",     "engine_id": 3, "name": "Hydraulic Press 03"},
    {"machineId": "pkg-04",     "engine_id": 4, "name": "Packaging Line 04"},
    {"machineId": "lathe-05",   "engine_id": 5, "name": "Lathe 05"},
    {"machineId": "wld-06",     "engine_id": 6, "name": "Welding Cell 06"},
    {"machineId": "asm-07",     "engine_id": 7, "name": "Assembly 07"},
    {"machineId": "conv-08",    "engine_id": 8, "name": "Conveyor 08"},
]

# Per-machine baseline so each one drifts around its own setpoint.
# Picked so a few machines hover near critical thresholds, producing
# occasional emergency alerts.
BASELINES = {
    "cnc-01":     {"temp": 62, "vib": 0.30, "power": 15, "speed": 1200, "rul": 180},
    "robotic-02": {"temp": 55, "vib": 0.22, "power": 12, "speed":  900, "rul": 220},
    "hyd-03":     {"temp": 78, "vib": 0.55, "power": 26, "speed":  600, "rul":  35},  # near critical
    "pkg-04":     {"temp": 48, "vib": 0.18, "power":  9, "speed":  450, "rul": 260},
    "lathe-05":   {"temp": 67, "vib": 0.41, "power": 18, "speed": 1100, "rul": 140},
    "wld-06":     {"temp": 82, "vib": 0.38, "power": 22, "speed":  300, "rul":  90},
    "asm-07":     {"temp": 51, "vib": 0.20, "power": 11, "speed":  800, "rul": 200},
    "conv-08":    {"temp": 44, "vib": 0.15, "power":  7, "speed":  350, "rul": 290},
}

# Internal state per machine — drifts cycle, RUL, and current values.
STATE = {
    m["machineId"]: {
        "cycle": random.randint(20, 300),
        "rul":   BASELINES[m["machineId"]]["rul"],
        "phase": random.random() * math.tau,  # for sin-wave drift
    }
    for m in MACHINES
}


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ Connected to {BROKER}:{PORT}, publishing on '{TOPIC}'")
    else:
        print(f"❌ Failed to connect, rc={rc}")


def build_payload(machine: dict) -> dict:
    mid = machine["machineId"]
    base = BASELINES[mid]
    st = STATE[mid]

    # Drift: small sine wave + Gaussian-ish noise for realism.
    st["phase"] += 0.05
    drift = math.sin(st["phase"])

    temp = base["temp"] + drift * 4 + random.uniform(-1.5, 1.5)
    vib = max(0.05, base["vib"] + drift * 0.06 + random.uniform(-0.03, 0.03))
    power = max(0.5, base["power"] + drift * 1.5 + random.uniform(-0.8, 0.8))
    speed = max(0.0, base["speed"] + drift * 30 + random.uniform(-25, 25))

    # RUL slowly decays. Reset to a fresh baseline when it hits zero
    # so the publisher can run indefinitely.
    st["cycle"] += 1
    st["rul"] = max(0.0, st["rul"] - random.uniform(0.1, 0.4))
    if st["rul"] < 1.0:
        st["rul"] = base["rul"]

    # Status: most of the time "running"; occasional injected fault
    # so Flutter's emergency rules fire visibly during a demo run.
    status = "running"
    if random.random() < 0.005:
        status = "fault"
    elif st["rul"] < 5:
        status = "fault"

    payload = {
        "machineId": mid,
        "engine_id": machine["engine_id"],   # legacy alias kept
        "current_cycle": st["cycle"],
        "predicted_rul": round(st["rul"], 2),
        "temperature": round(temp, 2),
        "vibration": round(vib, 3),
        "power": round(power, 2),
        "speed": round(speed, 1),
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),

        # Legacy NASA-style maps so the Flutter SHAP card / sensor list
        # keeps showing data for each machine.
        "current_sensor_readings": {
            "s_2":  round(temp, 2),
            "s_4":  round(vib, 3),
            "s_11": round(power, 2),
            "s_3":  round(temp * 0.8, 2),
            "s_7":  round(temp * 0.6, 2),
            "s_8":  round(speed * 0.01, 2),
            "s_12": round(power * 0.7, 2),
            "s_15": round(vib * 100, 2),
        },
        "ai_root_causes": {
            "s_2":  round(0.30 + abs(drift) * 0.10, 3),
            "s_4":  round(0.25 + abs(drift) * 0.05, 3),
            "s_11": round(0.20 + abs(drift) * 0.05, 3),
            "s_15": round(0.15, 3),
            "s_3":  round(0.10, 3),
        },
    }
    return payload


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    interval = 1.0 / max(0.1, PUBLISH_HZ)
    print(f"🚀 Multi-machine publisher started — {len(MACHINES)} machines @ {PUBLISH_HZ} Hz each")

    stopping = {"flag": False}
    def _stop(*_): stopping["flag"] = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while not stopping["flag"]:
            for m in MACHINES:
                payload = build_payload(m)
                client.publish(TOPIC, json.dumps(payload))
                # tiny pause between machines so they don't all batch
                time.sleep(interval / len(MACHINES))
    finally:
        client.loop_stop()
        client.disconnect()
        print("🏁 Publisher stopped.")


if __name__ == "__main__":
    main()

import os
import sys
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
from pathlib import Path
import time

# --- Configuration & Standards ---
# English comments only, no emojis in code.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)

# --- Path Setup ---
# Dynamically locate the project root
BASE_DIR = Path(__file__).resolve().parents[1]
AI_SRC_PATH = BASE_DIR / "01_AI_and_Data" / "src"
sys.path.insert(0, str(AI_SRC_PATH))

import shap_explainer

# --- Constants ---
BROKER = "127.0.0.1"
PORT = 1883
RAW_TOPIC = "digital_twin/raw_sensors"
AI_TOPIC = "digital_twin/engine_telemetry"

SEQ_LENGTH = 30
FEATURES_COUNT = 16
MAX_USEFUL_RUL = 125.0
FEATURE_COLS = [
    'setting_1', 'setting_2', 's_2', 's_3', 's_4', 's_7', 's_8', 's_9',
    's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21'
]

# --- Global State ---
model = None
scaler = None
explainer = None
# Dictionary to store sliding windows for multiple machines
machine_windows = {}

def load_assets():
    """Load trained models and initialize XAI explainer."""
    global model, scaler, explainer
    model_path = BASE_DIR / "01_AI_and_Data" / "saved_models" / "calibrated_model.keras"
    scaler_path = BASE_DIR / "01_AI_and_Data" / "saved_models" / "scalers" / "calibrated_scaler.gz"
    data_path = BASE_DIR / "01_AI_and_Data" / "data" / "raw" / "CMAPSSData" / "test_FD001.txt"

    print("Loading AI Assets...")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Initialize SHAP with a background sample
    bg_df = pd.read_csv(data_path, sep=r'\s+', header=None).iloc[:100]
    # Simple background initialization
    background_sample = np.zeros((10, SEQ_LENGTH, FEATURES_COUNT))
    explainer = shap_explainer.setup_shap_explainer(model, background_sample)
    print("AI Assets Loaded Successfully.")

def get_status(health):
    """Determine machine status string based on health score."""
    if health > 70: return "running"
    if health > 30: return "warning"
    return "fault"

# Add this to the global state section at the top of the file
last_publish_time = {}

def on_message(client, userdata, msg):
    global machine_windows, last_publish_time
    try:
        raw_payload = json.loads(msg.payload.decode())
        
        # Force the ID to match the Flutter app's expected catalog
        m_id = "cnc-01" 
        
        if m_id not in machine_windows:
            machine_windows[m_id] = []
            last_publish_time[m_id] = 0.0 # Initialize timer
            
        raw_features = [
            raw_payload['settings'][0], raw_payload['settings'][1],
            raw_payload['sensors'][1],  raw_payload['sensors'][2],
            raw_payload['sensors'][3],  raw_payload['sensors'][6],
            raw_payload['sensors'][7],  raw_payload['sensors'][8],
            raw_payload['sensors'][10], raw_payload['sensors'][11],
            raw_payload['sensors'][12], raw_payload['sensors'][13],
            raw_payload['sensors'][14], raw_payload['sensors'][16],
            raw_payload['sensors'][19], raw_payload['sensors'][20]
        ]
        
        scaled_feat = scaler.transform([raw_features])[0]
        machine_windows[m_id].append(scaled_feat)
        
        if len(machine_windows[m_id]) > SEQ_LENGTH:
            machine_windows[m_id].pop(0)
            
        if len(machine_windows[m_id]) == SEQ_LENGTH:
            current_time = time.time()
            
            current_window = np.array([machine_windows[m_id]], dtype=np.float32)
            
            prediction = model.predict(current_window, verbose=0)
            pred_rul = float(prediction[0][0])
            
            causes = shap_explainer.extract_fault_causes(explainer, current_window, FEATURE_COLS)
            
            health = min(max((pred_rul / MAX_USEFUL_RUL) * 100, 0.0), 100.0)
            status = get_status(health)
            
            final_payload = {
                "machineId": m_id,
                "engine_id": raw_payload.get('engine_id', 0),
                "current_cycle": raw_payload.get('cycle', 0),
                "predicted_rul": round(pred_rul, 2),
                "healthScore": round(health, 2),
                "status": status,
                "temperature": round(raw_payload['sensors'][1], 2), 
                "vibration": round(raw_payload['sensors'][3], 3),   
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "current_sensor_readings": {
                    feat: float(raw_features[i]) for i, feat in enumerate(FEATURE_COLS)
                },
                "ai_root_causes": causes
            }
            
            client.publish(AI_TOPIC, json.dumps(final_payload), qos=1, retain=False)
            print(f"Published Cycle: {raw_payload.get('cycle')} | Health: {round(health, 1)}%")

    except Exception as e:
        print(f"Inference Error: {e}")

def main():
    load_assets()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "AI_Inference_Publisher")
    client.on_message = on_message
    
    try:
        client.connect(BROKER, PORT, 60)
        client.subscribe(RAW_TOPIC)
        print(f"AI Engine connected to {BROKER}. Listening on {RAW_TOPIC}...")
        client.loop_forever()
    except KeyboardInterrupt:
        print("Stopping AI Engine...")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()
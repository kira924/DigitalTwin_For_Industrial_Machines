import os
import sys
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import paho.mqtt.client as mqtt

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import the SHAP logic module
import shap_explainer

# --- MQTT SETUP ---
BROKER_ADDRESS = "127.0.0.1"
PORT = 1883
TOPIC = "digital_twin/engine_telemetry"

def run_local_test():
    print("Starting Local Integration Test...")

    # 1. Setup dynamic paths based on project structure
    models_dir = os.path.join(parent_dir, 'saved_models')
    scaler_dir = os.path.join(parent_dir, 'saved_models')
    model_path = os.path.join(models_dir, 'calibrated_model.keras')
    scaler_path = os.path.join(scaler_dir, 'scalers' , 'calibrated_scaler.gz') 
    data_path = os.path.join(parent_dir, 'data', 'raw' , 'CMAPSSData' ,'test_FD001.txt')

    # 2. Load the trained Model and the Scaler
    print("Loading Model and Scaler...")
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return

    # 3. Load and clean the Test Data
    col_names = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                ['s_{}'.format(i) for i in range(1, 22)]
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=col_names)

    cols_to_drop = ['setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    df_clean = df.drop(columns=cols_to_drop)
    feature_cols = [c for c in df_clean.columns if c not in ['unit_nr', 'time_cycles']]

    # 4. Filter data for a specific engine
    engine_id = 34
    engine_data = df_clean[df_clean['unit_nr'] == engine_id].copy()

    # Scale features using the saved scaler
    engine_data[feature_cols] = scaler.transform(engine_data[feature_cols])
    data_matrix = engine_data[feature_cols].values

    # 5. Initialize SHAP Explainer
    print("Preparing SHAP Explainer...")
    seq_length = 30
    
    # Extract background data for SHAP
    background_windows = []
    for i in range(min(50, len(data_matrix) - seq_length)):
        background_windows.append(data_matrix[i:i+seq_length])
    background_data = np.array(background_windows, dtype=np.float32)

    explainer = shap_explainer.setup_shap_explainer(model, background_data)

    # --- Initialize MQTT Client ---
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "Python_Streamer")
    try:
        client.connect(BROKER_ADDRESS, PORT, 60)
        print("Successfully connected to the local MQTT Broker!")
    except Exception as e:
        print(f"MQTT Connection failed: {e}")
        return

    client.loop_start()

    # 6. Simulate Real-time Streaming and Prediction
    print("\n--- Starting Live Stream Simulation ---\n")
    
    try:
        for i in range(len(data_matrix) - seq_length + 1):
            current_window = np.array([data_matrix[i:i+seq_length]], dtype=np.float32)
            current_cycle = engine_data.iloc[i+seq_length-1]['time_cycles']

            rul_prediction = float(model.predict(current_window, verbose=0)[0][0])
            explanations = shap_explainer.extract_fault_causes(explainer, current_window, feature_cols)
            current_sensors = {feat: float(current_window[0, -1, idx]) for idx, feat in enumerate(feature_cols)}

            # Build the payload dictionary
            payload = {
                "engine_id": int(engine_id),
                "current_cycle": int(current_cycle),
                "predicted_rul": round(rul_prediction, 2),
                "current_sensor_readings": current_sensors,
                "ai_root_causes": explanations
            }

            # Publish strictly as JSON over MQTT
            client.publish(TOPIC, json.dumps(payload))
            print(f"Published Cycle {current_cycle} | Predicted RUL: {round(rul_prediction, 2)} to MQTT...")

            # Pause to mimic sensor delay
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        # Cleanly disconnect from MQTT
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    run_local_test()
import os
import sys
import json
import warnings
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import paho.mqtt.client as mqtt

# --- Warnings Suppressors ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

import shap_explainer

# --- MQTT SETUP ---
BROKER_ADDRESS = "127.0.0.1"
PORT = 1883
RAW_TOPIC = "digital_twin/raw_sensors"
AI_TOPIC = "digital_twin/engine_telemetry"

# Global variables for real-time processing
sliding_window = []
SEQ_LENGTH = 30
model = None
scaler = None
explainer = None
feature_cols = []

def load_background_data_for_shap(data_path):
    # Load a small chunk of historical data just to initialize SHAP
    col_names = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                ['s_{}'.format(i) for i in range(1, 22)]
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=col_names)

    cols_to_drop = ['setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    df_clean = df.drop(columns=cols_to_drop)
    
    global feature_cols
    feature_cols = [c for c in df_clean.columns if c not in ['unit_nr', 'time_cycles']]

    # Filter for a specific engine to extract a clean background sample
    engine_data = df_clean[df_clean['unit_nr'] == 34].copy()
    
    # Silence sklearn warning by passing a DataFrame with feature names
    df_engine_features = engine_data[feature_cols]
    engine_data[feature_cols] = scaler.transform(df_engine_features)
    
    data_matrix = engine_data[feature_cols].values

    background_windows = []
    for i in range(min(50, len(data_matrix) - SEQ_LENGTH)):
        background_windows.append(data_matrix[i:i+SEQ_LENGTH])
        
    return np.array(background_windows, dtype=np.float32)

def on_message(client, userdata, msg):
    global sliding_window

    try:
        payload = json.loads(msg.payload.decode())
        
        # Extract features from the Webots payload to match training features exactly
        # Note: Webots arrays are 0-indexed, so s_2 is sensors[1], s_21 is sensors[20]
        raw_features = {
            'setting_1': payload['settings'][0],
            'setting_2': payload['settings'][1],
            's_2': payload['sensors'][1],
            's_3': payload['sensors'][2],
            's_4': payload['sensors'][3],
            's_7': payload['sensors'][6],
            's_8': payload['sensors'][7],
            's_9': payload['sensors'][8],
            's_11': payload['sensors'][10],
            's_12': payload['sensors'][11],
            's_13': payload['sensors'][12],
            's_14': payload['sensors'][13],
            's_15': payload['sensors'][14],
            's_17': payload['sensors'][16],
            's_20': payload['sensors'][19],
            's_21': payload['sensors'][20],
        }

        # Convert dictionary to a list in the exact order of feature_cols
        current_features_list = [raw_features[feat] for feat in feature_cols]

        # Scale the incoming features (Fix warning by using DataFrame)
        df_features = pd.DataFrame([current_features_list], columns=feature_cols)
        scaled_features = scaler.transform(df_features)[0]

        # Append to the sliding window buffer
        sliding_window.append(scaled_features)

        # Maintain the window size to strictly equal SEQ_LENGTH
        if len(sliding_window) > SEQ_LENGTH:
            sliding_window.pop(0)

        # Once we have collected enough cycles, start predicting
        if len(sliding_window) == SEQ_LENGTH:
            current_window = np.array([sliding_window], dtype=np.float32)
            current_cycle = payload['cycle']
            engine_id = payload['engine_id']

            # 1. AI Inference
            rul_prediction = float(model.predict(current_window, verbose=0)[0][0])
            
            # 2. Explainability
            explanations = shap_explainer.extract_fault_causes(explainer, current_window, feature_cols)
            
            # 3. Extract scaled sensors for the dashboard
            current_sensors = {feat: float(scaled_features[idx]) for idx, feat in enumerate(feature_cols)}

            # Build final payload
            out_payload = {
                "engine_id": engine_id,
                "current_cycle": current_cycle,
                "predicted_rul": round(rul_prediction, 2),
                "current_sensor_readings": current_sensors,
                "ai_root_causes": explanations
            }

            # Publish the AI results back to MQTT
            client.publish(AI_TOPIC, json.dumps(out_payload))
            print(f"Processed Cycle {current_cycle} | Predicted RUL: {round(rul_prediction, 2)} | Published to UI & Twin")

    except Exception as e:
        print(f"Error processing incoming MQTT message: {e}")

def run_edge_ai_node():
    global model, scaler, explainer

    print("Starting AI Edge Engine...")

    # Setup paths
    models_dir = os.path.join(parent_dir, 'saved_models')
    scaler_dir = os.path.join(parent_dir, 'saved_models')
    model_path = os.path.join(models_dir, 'calibrated_model.keras')
    scaler_path = os.path.join(scaler_dir, 'scalers' , 'calibrated_scaler.gz') 
    data_path = os.path.join(parent_dir, 'data', 'raw' , 'CMAPSSData' ,'test_FD001.txt')

    print("Loading Model and Scaler...")
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return

    print("Preparing SHAP Explainer...")
    background_data = load_background_data_for_shap(data_path)
    explainer = shap_explainer.setup_shap_explainer(model, background_data)

    # Initialize MQTT Client and attach the listener
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "AI_Inference_Engine")
    client.on_message = on_message

    try:
        client.connect(BROKER_ADDRESS, PORT, 60)
        client.subscribe(RAW_TOPIC)
        print("\nSuccessfully connected to the local MQTT Broker!")
        print(f"Listening for raw Webots telemetry on topic: {RAW_TOPIC}")
        print("Waiting for physical simulation to start...\n")
        
        # Block the script and keep it listening forever
        client.loop_forever()

    except KeyboardInterrupt:
        print("\nAI Engine stopped by user.")
    except Exception as e:
        print(f"MQTT Connection failed: {e}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    run_edge_ai_node()

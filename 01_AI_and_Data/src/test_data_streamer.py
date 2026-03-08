import os
import sys
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import the SHAP logic module we created earlier
import shap_explainer

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

    # Drop the exact same columns we dropped during training
    cols_to_drop = ['setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    df_clean = df.drop(columns=cols_to_drop)
    feature_cols = [c for c in df_clean.columns if c not in ['unit_nr', 'time_cycles']]

    # 4. Filter data for a specific engine (e.g., Engine 34)
    engine_id = 34
    engine_data = df_clean[df_clean['unit_nr'] == engine_id].copy()

    # Scale features using the saved scaler
    engine_data[feature_cols] = scaler.transform(engine_data[feature_cols])
    data_matrix = engine_data[feature_cols].values

    # 5. Initialize SHAP Explainer
    print("Preparing SHAP Explainer...")
    seq_length = 30
    
    # Extract background data for SHAP (first 50 windows)
    background_windows = []
    for i in range(min(50, len(data_matrix) - seq_length)):
        background_windows.append(data_matrix[i:i+seq_length])
    background_data = np.array(background_windows, dtype=np.float32)

    explainer = shap_explainer.setup_shap_explainer(model, background_data)

    # 6. Simulate Real-time Streaming and Prediction
    print("\n--- Starting Live Stream Simulation ---\n")
    
    # Loop through the data to simulate time passing cycle by cycle
    for i in range(len(data_matrix) - seq_length + 1):
        # Extract the current sliding window of shape [1, 30, 16]
        current_window = np.array([data_matrix[i:i+seq_length]], dtype=np.float32)
        current_cycle = engine_data.iloc[i+seq_length-1]['time_cycles']

        # AI Prediction: Get the RUL
        rul_prediction = float(model.predict(current_window, verbose=0)[0][0])

        # AI Explainability: Get SHAP root causes
        explanations = shap_explainer.extract_fault_causes(explainer, current_window, feature_cols)

        # Extract current sensor values for the dashboard (the last row in the window)
        current_sensors = {feat: float(current_window[0, -1, idx]) for idx, feat in enumerate(feature_cols)}

        # 7. Build the final JSON Payload to match mobile app expectations
        payload = {
            "engine_id": int(engine_id),
            "current_cycle": int(current_cycle),
            "predicted_rul": round(rul_prediction, 2),
            "current_sensor_readings": current_sensors,
            "ai_root_causes": explanations
        }

        # Convert dictionary to a formatted JSON string
        json_output = json.dumps(payload, indent=4)

        # Print to console for the developer to inspect
        print(f"Time Cycle: {current_cycle} | AI Predicted RUL: {round(rul_prediction, 2)}")
        print(f"JSON Payload (Ready for MQTT):\n{json_output}\n")
        print("-" * 60)

        # Pause for 1 second to mimic actual machine sensor delays
        time.sleep(1)

if __name__ == "__main__":
    run_local_test()
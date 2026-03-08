import os
import sys
import numpy as np
import tensorflow as tf
import shap

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def setup_shap_explainer(model, background_data):
    # Initialize SHAP GradientExplainer for the Keras LSTM model
    # background_data is a small sample (e.g., 100 windows) from your training set
    # It acts as a baseline to compare new predictions against
    print("Initializing SHAP GradientExplainer...")
    explainer = shap.GradientExplainer(model, background_data)
    return explainer

def extract_fault_causes(explainer, input_window, feature_names):
    # Calculate SHAP values for the current sliding window
    shap_values = explainer.shap_values(input_window)
    
    # Handle SHAP output format depending on the exact SHAP version
    if isinstance(shap_values, list):
        shap_array = shap_values[0]
    else:
        shap_array = shap_values
        
    # Extract the matrix for the single batch
    abs_shap_matrix = np.abs(shap_array[0])
    
    # Average the SHAP values across the 30 time steps
    mean_impact_per_feature = np.mean(abs_shap_matrix, axis=0)
    
    # FIX: Flatten the array to remove any extra dimensions like (16, 1) -> (16,)
    mean_impact_per_feature = np.squeeze(mean_impact_per_feature)
    
    # Convert raw SHAP impacts into human-readable percentages
    total_impact = np.sum(mean_impact_per_feature)
    
    if total_impact == 0:
        percentages = np.zeros(len(feature_names))
    else:
        percentages = (mean_impact_per_feature / total_impact) * 100
        
    # Map each feature name to its calculated percentage and cast to float
    impact_dict = {
        name: round(float(pct), 2) 
        for name, pct in zip(feature_names, percentages)
    }
    
    # Sort the dictionary by percentage in descending order
    sorted_impact = dict(sorted(impact_dict.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_impact

# --- TEST THE LOGIC ---
if __name__ == "__main__":
    # 1. Dummy feature names based on your remaining 16 sensors
    cols = ['setting_1', 'setting_2', 's_2', 's_3', 's_4', 's_7', 's_8', 's_9', 
            's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    
    # 2. Mocking a loaded model and dummy data for testing purposes
    # In production, replace this with: tf.keras.models.load_model('calibrated_model.keras')
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(30, 16)),
        tf.keras.layers.Dense(1)
    ])
    
    # 3. Create dummy background data (e.g., 50 random windows) and initialize explainer
    # In production, use a subset of your scaled training data
    dummy_background = np.random.rand(50, 30, 16).astype(np.float32)
    explainer = setup_shap_explainer(dummy_model, dummy_background)
    
    # 4. Simulate reading a live window from MQTT (1 batch, 30 steps, 16 features)
    live_window = np.random.rand(1, 30, 16).astype(np.float32)
    
    # 5. Extract explanations
    causes = extract_fault_causes(explainer, live_window, cols)
    
    print("\nRoot Cause Analysis (Top 3):")
    top_3 = list(causes.items())[:3]
    for sensor, impact in top_3:
        print(f"Sensor {sensor}: {impact}% responsibility for RUL drop")
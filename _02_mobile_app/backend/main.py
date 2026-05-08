import os
import sys
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal logic imports
# Assuming shap_explainer.py is in the same directory or accessible via path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adding the AI source directory to path to access shap_explainer
sys.path.append(os.path.join(current_dir, "../../01_AI_and_Data/src"))
import shap_explainer

app = FastAPI(title="Industrial Digital Twin Production API")

# Enable CORS for Flutter Web (Edge) and other clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration Constants
MAX_USEFUL_RUL = 125.0
SEQ_LENGTH = 30
FEATURES_COUNT = 16
FEATURE_COLS = [
    'setting_1', 'setting_2', 's_2', 's_3', 's_4', 's_7', 's_8', 's_9',
    's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21'
]

# Assets Paths
BASE_PROJECT_PATH = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_PROJECT_PATH / "01_AI_and_Data" / "saved_models" / "calibrated_model.keras"
SCALER_PATH = BASE_PROJECT_PATH / "01_AI_and_Data" / "saved_models" / "scalers" / "calibrated_scaler.gz"
DATA_PATH = BASE_PROJECT_PATH / "01_AI_and_Data" / "data" / "raw" / "CMAPSSData" / "test_FD001.txt"

# Global model and utility holders
model = None
scaler = None
explainer = None

class PredictionInput(BaseModel):
    machine_id: str
    engine_id: int
    current_cycle: int
    series_data: list[list[float]] # Expected shape: (30, 16)

@app.on_event("startup")
def load_production_assets():
    """Load AI models, scalers and initialize SHAP explainer on startup."""
    global model, scaler, explainer
    try:
        # Load the Keras LSTM model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load the fitted StandardScaler
        scaler = joblib.load(SCALER_PATH)
        
        # Initialize SHAP for Explainability
        # Loading a small background sample from training data for SHAP initialization
        import pandas as pd
        raw_data = pd.read_csv(DATA_PATH, sep=r'\s+', header=None).iloc[:500]
        # Preprocessing background data to match model expectations
        # (Simplified for this production script)
        background_sample = np.zeros((10, SEQ_LENGTH, FEATURES_COUNT)) 
        explainer = shap_explainer.setup_shap_explainer(model, background_sample)
        
        print("Production assets loaded successfully.")
    except Exception as e:
        print(f"Error loading assets: {str(e)}")

def determine_machine_status(health_score: float) -> str:
    """Classify machine state based on health percentage."""
    if health_score > 70:
        return "running"
    elif health_score > 30:
        return "warning"
    else:
        return "fault"

@app.post("/predict")
async def predict_telemetry(data: PredictionInput):
    """Process time-series data and return explainable RUL insights."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Inference engine not ready.")

    try:
        # 1. Validation and Preparation
        input_array = np.array(data.series_data, dtype=np.float32)
        if input_array.shape != (SEQ_LENGTH, FEATURES_COUNT):
            raise ValueError(f"Invalid input shape. Expected (30, 16), got {input_array.shape}")

        # 2. Scaling (Applying the transformation used during training)
        scaled_data = scaler.transform(input_array)
        scaled_window = np.expand_dims(scaled_data, axis=0) # Shape: (1, 30, 16)

        # 3. AI Inference (RUL Prediction)
        prediction = model.predict(scaled_window, verbose=0)
        predicted_rul = float(prediction[0][0])
        
        # 4. Explainability (SHAP Values)
        # Extracting top causes for the current prediction
        root_causes = shap_explainer.extract_fault_causes(explainer, scaled_window, FEATURE_COLS)

        # 5. Business Logic (Health and Status)
        health_score = (predicted_rul / MAX_USEFUL_RUL) * 100
        health_score = min(max(health_score, 0.0), 100.0) # Clamp between 0-100
        status = determine_machine_status(health_score)

        # 6. Response Construction (Matching Flutter's EngineTelemetryPayload)
        return {
            "machineId": data.machine_id,
            "engine_id": data.engine_id,
            "current_cycle": data.current_cycle,
            "predicted_rul": round(predicted_rul, 2),
            "healthScore": round(health_score, 2),
            "status": status,
            "current_sensor_readings": {
                feat: float(input_array[-1][idx]) for idx, feat in enumerate(FEATURE_COLS)
            },
            "ai_root_causes": root_causes,
            "timestamp": tf.timestamp().numpy() # Simple timestamp or use datetime
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
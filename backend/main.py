import joblib
import numpy as np
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow import keras

app = FastAPI(title="Digital Twin Prediction API")

# ✅ حل مشكلة CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# المسارات
MODEL_PATH = Path(__file__).with_name("calibrated_model.keras")
SCALER_PATH = Path(__file__).with_name("calibrated_scaler")

model = None
scaler = None

class InputData(BaseModel):
    series_data: list[list[float]]

@app.on_event("startup")
def load_assets() -> None:
    global model, scaler
    try:
        # تحميل الموديل
        model = keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        # تحميل السكيلر
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            print("✅ Scaler loaded successfully!")
        else:
            print("⚠️ Scaler file not found! Predictions might be inaccurate.")

    except Exception as e:
        print(f"❌ Error during startup: {e}")

@app.get("/")
def healthcheck() -> dict:
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict")
async def predict(data: InputData) -> dict:
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # 1. تحويل البيانات لمصفوفة Numpy
        arr = np.array(data.series_data, dtype=np.float32) # الشكل المتوقع (30, 16)

        if arr.shape != (30, 16):
            raise HTTPException(
                status_code=400,
                detail=f"Expected shape (30,16) but got {arr.shape}",
            )

        # 2. تطبيق الـ Scaling (مهم جداً لتحريك الـ RUL)
        if scaler is not None:
            # السكيلر يتوقع شكل 2D (Sample, Features)
            arr_reshaped = arr.reshape(-1, 16)
            arr_scaled = scaler.transform(arr_reshaped)
            arr = arr_scaled.reshape(30, 16)
            print("✨ Data scaled successfully")

        # 3. تحويل الشكل لما يتوقعه الموديل (إضافة بعد الـ Batch)
        arr_final = np.expand_dims(arr, axis=0) # الشكل يصبح (1, 30, 16)

        # 4. التنبؤ
        prediction = model.predict(arr_final, verbose=0)
        value = float(np.asarray(prediction).reshape(-1)[0])

        print(f"🤖 Prediction Result: {value}")

        return {
            "prediction": value,
            "status": "success",
        }

    except Exception as e:
        print("❌ Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
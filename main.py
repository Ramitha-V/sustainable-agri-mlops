from fastapi import FastAPI
from fastapi.responses import FileResponse # Import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Initialize FastAPI app
app = FastAPI()

# --- Model Loading ---
# Ensure the model and columns are loaded correctly relative to this script's location
try:
    model_path = "saved_models/model.joblib"
    train_cols_path = "data/processed/X_train.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(train_cols_path):
        raise FileNotFoundError("Model or training columns file not found. Have you run 'dvc pull'?")

    model = joblib.load(model_path)
    train_cols = list(pd.read_csv(train_cols_path).columns)
    print("Model and training columns loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    train_cols = []


# --- Pydantic Model for Input Validation ---
class CropInput(BaseModel):
    Crop: str
    Crop_Year: int
    Season: str
    State: str
    Area: float
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float


# --- API Endpoints ---

# This endpoint now correctly serves your HTML file as the root page
@app.get("/")
async def read_index():
    # Use FileResponse to serve the index.html file
    return FileResponse("index.html")


@app.post("/predict")
async def predict_yield(data: CropInput):
    if not model or not train_cols:
        return {"error": "Model not loaded properly. Check server logs."}

    try:
        input_df = pd.DataFrame([data.dict()])
        input_encoded = pd.get_dummies(input_df)
        
        # Align columns with the training data
        final_df = pd.DataFrame(columns=train_cols)
        final_df = pd.concat([final_df, input_encoded]).fillna(0)
        final_df = final_df[train_cols] # Ensure column order is the same

        prediction = model.predict(final_df)[0]
        return {"predicted_yield": prediction}

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

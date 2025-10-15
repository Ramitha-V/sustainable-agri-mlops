# main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize FastAPI app
app = FastAPI()

# --- Model Loading ---
# Load the pre-trained model and column information
try:
    model = joblib.load("saved_models/model.joblib")
    # Load the training columns to ensure input has the same features
    train_cols = list(pd.read_csv("data/processed/X_train.csv").columns)
    print("Model and training columns loaded successfully.")
except FileNotFoundError:
    print("Error: Model or training columns file not found. Make sure 'model.joblib' and 'X_train.csv' exist.")
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
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Read the content of index.html
    with open("index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict")
async def predict_yield(data: CropInput):
    if not model or not train_cols:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    try:
        # Convert input data to a pandas DataFrame
        input_df = pd.DataFrame([data.dict()])

        # One-hot encode the categorical features
        # Create a DataFrame with all possible columns from training set, filled with 0
        input_encoded = pd.DataFrame(columns=train_cols)
        input_encoded = pd.concat([input_encoded, pd.get_dummies(input_df)], ignore_index=True).fillna(0)
        
        # Ensure all columns from training are present
        for col in train_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reorder columns to match model's training order
        input_encoded = input_encoded[train_cols]

        # Make prediction
        prediction = model.predict(input_encoded)[0]

        return {"predicted_yield": prediction}

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# Serve static files (if you have a CSS or JS folder)
# app.mount("/static", StaticFiles(directory="static"), name="static")

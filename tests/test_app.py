# tests/test_app.py
import pytest
from fastapi.testclient import TestClient
# Import app from main.py in the root
from main import app
import joblib
import pandas as pd
import os
from unittest.mock import patch # Import patch

# --- Mocking Dependencies ---

# Sample processed data matching expected columns after get_dummies
# Use the same SAMPLE_X_TRAIN as before, its columns represent train_cols
SAMPLE_X_TRAIN_COLS_DF = pd.DataFrame({
    'Crop_Year': [2000, 2001, 2000], 'Area': [100.0, 120.0, 150.0],
    'Annual_Rainfall': [800.0, 850.0, 600.0], 'Fertilizer': [1000.0, 1100.0, 1200.0],
    'Pesticide': [100.0, 120.0, 150.0], 'Season_Rabi': [0, 0, 1],
    'State_Punjab': [1, 0, 1], 'Crop_Wheat': [0, 0, 1], 'Crop_Maize': [0, 1, 0]
})
# Extract column names list as expected by main.py
DUMMY_TRAIN_COLS = SAMPLE_X_TRAIN_COLS_DF.columns.tolist()

# Create a dummy model class
class DummyModel:
    def predict(self, data):
        # Ensure predict returns a list or array
        return [4.5] * len(data)

@pytest.fixture(scope="module")
def test_client(tmp_path_factory):
    """Creates a TestClient instance, patches globals, and sets up dummy files."""
    temp_dir = tmp_path_factory.mktemp("app_test_files_main")
    templates_dir = temp_dir / "templates"; templates_dir.mkdir()

    dummy_model_instance = DummyModel()

   
    with patch('main.model', dummy_model_instance), \
         patch('main.train_cols', DUMMY_TRAIN_COLS), \
         patch('main.FileResponse') as MockFileResponse: 
        with TestClient(app) as client:
            yield client
    
def test_read_root(test_client):
    """Test the root endpoint (checks if it tries to return a file)."""
    
    response = test_client.get("/")
    assert response.status_code == 200
    

def test_predict_yield_success(test_client):
    """Test the /predict endpoint with valid data."""
    test_payload = {
        "Crop": "Rice", "Crop_Year": 2024, "Season": "Kharif",
        "State": "Punjab", "Area": 50000, "Annual_Rainfall": 850.5,
        "Fertilizer": 12000, "Pesticide": 850
    }
    response = test_client.post("/predict", json=test_payload)
    assert response.status_code == 200
    result = response.json()
    assert "predicted_yield" in result
    assert result["predicted_yield"] == 4.5 

def test_predict_yield_missing_data(test_client):
    """Test the /predict endpoint with incomplete data."""
    test_payload = {"Crop": "Rice"} 
    response = test_client.post("/predict", json=test_payload)
    
    assert response.status_code == 422 
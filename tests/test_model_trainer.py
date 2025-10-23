# tests/test_model_trainer.py (or rename to test_train.py)
import pytest
import pandas as pd
import os
import subprocess
import yaml
import json # Keep json if you add metrics later
import joblib
import sys

SAMPLE_X_TRAIN = pd.DataFrame({
    'Crop_Year': [2000, 2001, 2000], 'Area': [100.0, 120.0, 150.0],
    'Annual_Rainfall': [800.0, 850.0, 600.0], 'Fertilizer': [1000.0, 1100.0, 1200.0],
    'Pesticide': [100.0, 120.0, 150.0], 'Season_Rabi': [0, 0, 1],
    'State_Punjab': [1, 0, 1], 'Crop_Wheat': [0, 0, 1], 'Crop_Maize': [0, 1, 0]
})
SAMPLE_Y_TRAIN = pd.DataFrame({'Yield': [5.0, 4.58, 4.0]})
# Add SAMPLE_X_TEST and SAMPLE_Y_TEST if your train.py evaluates on test data
# SAMPLE_X_TEST = ...
# SAMPLE_Y_TEST = ...

@pytest.fixture
def setup_trainer_environment(tmp_path):
    data_dir = tmp_path / "data"; processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)
    # Define where the script expects to save the model (relative to cwd=tmp_path)
    model_output_dir = tmp_path / "saved_models" # Script creates this dir based on params
    # Model path relative to tmp_path, matching the param value below
    relative_model_path = "saved_models/model.joblib"

    SAMPLE_X_TRAIN.to_csv(processed_dir / "X_train.csv", index=False)
    SAMPLE_Y_TRAIN.to_csv(processed_dir / "y_train.csv", index=False)
    # SAMPLE_X_TEST.to_csv(processed_dir / "X_test.csv", index=False)
    # SAMPLE_Y_TEST.to_csv(processed_dir / "y_test.csv", index=False)

    params_path = tmp_path / "params.yaml"
    # Create params.yaml with the 'train' section expected by train.py
    test_params = {
        'data_preprocessing': {}, # Dummy entry if needed
        'train': { # Correct section name
             'n_estimators': 5, # Small values for fast tests
             'max_depth': 2,
             'min_samples_split': 2,
             'random_state': 42,
             'model_path': relative_model_path # Path where train.py saves the model
        }
    }
    with open(params_path, 'w') as f: yaml.dump(test_params, f)
    # Return the *full* expected model path for assertion
    return {"tmp_path": tmp_path, "expected_model_path": tmp_path / relative_model_path}

def test_model_trainer_script(setup_trainer_environment):
    tmp_path = setup_trainer_environment["tmp_path"]
    expected_model_path = setup_trainer_environment["expected_model_path"]

    # Calculate script path relative to this test file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, '..', 'src', 'train.py') # Use correct script name
    script_path = os.path.normpath(script_path)

    if not os.path.exists(script_path):
        pytest.fail(f"Script not found at calculated path: {script_path}")

    try:
        # Run train.py using the venv python, NO extra arguments
        subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=tmp_path, # Run from temp dir so script finds data/processed/ and params.yaml
            capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        error_message = f"Script failed with exit code {e.returncode}.\n"
        error_message += f"--- STDOUT ---\n{e.stdout}\n"
        error_message += f"--- STDERR ---\n{e.stderr}\n"
        pytest.fail(error_message)
    except FileNotFoundError:
         pytest.fail(f"Could not find python executable: {sys.executable} or script {script_path}")

    # --- Assert ---
    # Check if the model file was created at the path specified in params.yaml
    assert os.path.exists(expected_model_path), f"Model file not found at {expected_model_path}"
    # Remove metrics check as train.py doesn't create metrics

    # Check if the saved model can be loaded
    try:
        loaded_model = joblib.load(expected_model_path)
        assert hasattr(loaded_model, 'predict')
    except Exception as e:
        pytest.fail(f"Failed to load saved model: {e}")
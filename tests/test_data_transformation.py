# tests/test_data_transformation.py
import pytest
import pandas as pd
import os
import subprocess
import yaml
import sys

SAMPLE_RAW_DATA = pd.DataFrame({
    'Crop': ['Rice', 'Wheat', 'Rice', 'Maize'], 'Crop_Year': [2000, 2000, 2001, 2001],
    'Season': ['Kharif', 'Rabi', 'Kharif', 'Kharif'], 'State': ['Punjab', 'Punjab', 'Haryana', 'Haryana'],
    'Area': [100.0, 150.0, 120.0, 80.0], 'Production': [500.0, 600.0, 550.0, 400.0],
    'Annual_Rainfall': [800.0, 600.0, 850.0, 900.0], 'Fertilizer': [1000.0, 1200.0, 1100.0, 900.0],
    'Pesticide': [100.0, 150.0, 120.0, 80.0], 'Yield': [5.0, 4.0, 4.58, 5.0]
})

@pytest.fixture
def setup_test_environment(tmp_path):
    # Create directory structure expected by the script
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    processed_dir = data_dir / "processed" # Script will create this one

    # Save sample raw data to the 'raw' subdirectory
    raw_data_path = raw_dir / "crop_yield.csv"
    SAMPLE_RAW_DATA.to_csv(raw_data_path, index=False)

    # Create params.yaml with the correct structure
    params_path = tmp_path / "params.yaml"
    test_params = {
        'data_preprocessing': { # Correct section name
            'test_size': 0.25,
            'random_state': 42 # Add random_state used by script
        },
        'train': {} # Dummy train section if needed by other scripts called indirectly
    }
    with open(params_path, 'w') as f: yaml.dump(test_params, f)
    return {"tmp_path": tmp_path, "processed_dir": processed_dir}

def test_data_transformation_script(setup_test_environment):
    tmp_path = setup_test_environment["tmp_path"]
    processed_dir = setup_test_environment["processed_dir"]

    # Calculate script path relative to this test file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, '..', 'src', 'data_preprocessing.py')
    script_path = os.path.normpath(script_path)

    if not os.path.exists(script_path):
        pytest.fail(f"Script not found at calculated path: {script_path}")

    try:
        # Run the script using the virtual environment's python
        subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=tmp_path, # Run from temp dir so it finds data/raw/ and params.yaml
            capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        error_message = f"Script failed with exit code {e.returncode}.\n"
        error_message += f"--- STDOUT ---\n{e.stdout}\n"
        error_message += f"--- STDERR ---\n{e.stderr}\n"
        pytest.fail(error_message)
    except FileNotFoundError:
         pytest.fail(f"Could not find python executable: {sys.executable} or script {script_path}")

    # Assertions remain similar, checking output files in data/processed
    assert os.path.exists(processed_dir / "X_train.csv")
    assert os.path.exists(processed_dir / "X_test.csv")
    assert os.path.exists(processed_dir / "y_train.csv")
    assert os.path.exists(processed_dir / "y_test.csv")
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    assert len(X_train) == 3
    assert len(X_test) == 1
    assert X_train.isnull().sum().sum() == 0
    assert 'Crop' not in X_train.columns
    assert any(col.startswith("State_") for col in X_train.columns)
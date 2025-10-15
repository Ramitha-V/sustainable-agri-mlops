# src/evaluate.py
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import yaml
import joblib
import json

def evaluate_model():
    train_params = yaml.safe_load(open("params.yaml"))["train"]
    eval_params = yaml.safe_load(open("params.yaml"))["evaluate"]

    # Load test data
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # Load the model
    model = joblib.load(train_params['model_path'])

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    # Save metrics to a JSON file
    with open(eval_params['metrics_file'], 'w') as f:
        json.dump({
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2
        }, f, indent=4)

if __name__ == "__main__":
    evaluate_model()
# src/train.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yaml
import joblib
import os

def train_model():
    params = yaml.safe_load(open("params.yaml"))["train"]
    
    # Load training data
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    # Initialize and train the model
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        random_state=params['random_state'],
        n_jobs=-1 # Use all available cores
    )
    
    model.fit(X_train, y_train)
    
    # Save the trained model
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, params['model_path'])

if __name__ == "__main__":
    train_model()

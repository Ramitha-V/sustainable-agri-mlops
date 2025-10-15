# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def preprocess_data():
    params = yaml.safe_load(open("params.yaml"))["data_preprocessing"]
    
    # Load data
    df = pd.read_csv("data/raw/crop_yield.csv")
    
    # Simple cleaning: drop rows with missing values
    df.dropna(inplace=True)

    # Feature Engineering (optional, can be expanded)
    # df['Fertilizer_Pesticide_Ratio'] = df['Fertilizer'] / (df['Pesticide'] + 1e-6)

    # One-Hot Encode categorical features
    categorical_cols = ['Crop', 'Season', 'State']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Define features (X) and target (y)
    X = df_encoded.drop(columns=['Yield', 'Production']) # Production is highly correlated with Yield
    y = df_encoded['Yield']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']
    )

    # Create processed data directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    # Save processed data
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

if __name__ == "__main__":
    preprocess_data()

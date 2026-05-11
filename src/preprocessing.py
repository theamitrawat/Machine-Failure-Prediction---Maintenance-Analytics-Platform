# src/preprocessing.py

import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Mapping from raw CSV column names (with brackets/units) to clean names
# that match what the saved model and feature_columns.pkl expect
COLUMN_RENAME_MAP = {
    'Air temperature [K]': 'Air temperature K',
    'Process temperature [K]': 'Process temperature K',
    'Rotational speed [rpm]': 'Rotational speed rpm',
    'Torque [Nm]': 'Torque Nm',
    'Tool wear [min]': 'Tool wear min',
}

def load_and_preprocess(path, save_encoder_path=None, save_feature_cols_path=None):
    """
    Load dataset and preprocess it for ML training.

    Args:
        path (str): Path to dataset CSV
        save_encoder_path (str, optional): If provided, saves the LabelEncoder for 'Type' here
        save_feature_cols_path (str, optional): If provided, saves the feature column list here

    Returns:
        X (DataFrame): Feature dataframe with clean column names
        y (Series): Target variable ('Machine failure')
    """
    df = pd.read_csv(path)

    # Drop unnecessary ID columns
    df = df.drop(['UDI', 'Product ID'], axis=1, errors='ignore')

    # Rename columns: strip bracket/unit notation to match saved model expectations
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Encode categorical column 'Type' (M, L, H -> numeric)
    if 'Type' in df.columns:
        le = LabelEncoder()
        df['Type'] = le.fit_transform(df['Type'])

        # Save encoder if path provided so predict.py can use consistent encoding
        if save_encoder_path:
            os.makedirs(os.path.dirname(save_encoder_path), exist_ok=True)
            joblib.dump(le, save_encoder_path)

    # Drop rows with missing values
    df = df.dropna()

    # Separate features and target
    X = df.drop('Machine failure', axis=1)
    y = df['Machine failure']

    # Save feature column list if path provided
    if save_feature_cols_path:
        os.makedirs(os.path.dirname(save_feature_cols_path), exist_ok=True)
        joblib.dump(X.columns.tolist(), save_feature_cols_path)

    return X, y

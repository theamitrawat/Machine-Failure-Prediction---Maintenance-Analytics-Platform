# src/utils.py

import joblib
import pandas as pd


def load_model(path):
    """Load a trained ML model from a joblib file."""
    return joblib.load(path)


def load_feature_columns(path):
    """Load the saved list of feature column names from a joblib file."""
    return joblib.load(path)


def load_dataset(path):
    """Load a CSV dataset into a pandas DataFrame."""
    return pd.read_csv(path)

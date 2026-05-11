# src/train_model.py

import os
import sys
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Allow running from project root: python src/train_model.py
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_and_preprocess


def train_xgboost(dataset_path, model_save_path,
                  encoder_save_path=None, feature_cols_save_path=None):
    """
    Train an XGBoost classifier for machine failure prediction.

    Args:
        dataset_path (str): Path to the raw CSV dataset
        model_save_path (str): Where to save the trained model (.pkl)
        encoder_save_path (str, optional): Where to save the LabelEncoder for 'Type'
        feature_cols_save_path (str, optional): Where to save the feature column list

    Returns:
        xgb_model: Trained XGBClassifier
    """
    # Load and preprocess data; save encoder + feature columns alongside model
    X, y = load_and_preprocess(
        dataset_path,
        save_encoder_path=encoder_save_path,
        save_feature_cols_path=feature_cols_save_path,
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost model — scale_pos_weight handles class imbalance without extra deps
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count if pos_count > 0 else 1

    xgb_model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
    )

    # Train model
    xgb_model.fit(X_train, y_train)

    # Evaluate
    y_pred = xgb_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(xgb_model, model_save_path)
    print(f"Model saved at {model_save_path}")

    return xgb_model


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_xgboost(
        dataset_path=os.path.join(BASE_DIR, "data", "machine_failure_prediction_dataset.csv"),
        model_save_path=os.path.join(BASE_DIR, "models", "xgboost_machine_failure_model.pkl"),
        encoder_save_path=os.path.join(BASE_DIR, "models", "label_encoder.pkl"),
        feature_cols_save_path=os.path.join(BASE_DIR, "models", "feature_columns.pkl"),
    )

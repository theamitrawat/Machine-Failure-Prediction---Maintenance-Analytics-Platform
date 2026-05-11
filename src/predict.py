# src/predict.py

import numpy as np
import pandas as pd


def predict_machine_status(model, input_data, feature_columns):
    """
    Predict machine failure and provide maintenance analytics.

    Args:
        model: Trained XGBoost model
        input_data (list or array): Values in the same order as feature_columns.
            Expected order when using the standard 11-feature model:
            [Type (encoded), Air temperature K, Process temperature K,
             Rotational speed rpm, Torque Nm, Tool wear min,
             TWF, HDF, PWF, OSF, RNF]
        feature_columns (list): Column names the model was trained on
                                (loaded from models/feature_columns.pkl)

    Returns:
        dict: Prediction, probability, health score, risk, maintenance days,
              and recommendations
    """
    # --- Input validation ---
    if input_data is None or len(input_data) != len(feature_columns):
        raise ValueError(
            f"input_data must have {len(feature_columns)} values "
            f"matching feature_columns, got {len(input_data) if input_data else 0}."
        )

    # Wrap in a named DataFrame so XGBoost gets the correct column order/names
    sample = pd.DataFrame([input_data], columns=feature_columns)

    # --- Prediction & probability ---
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0]

    failure_prob = float(prob[1])

    # --- Health Score (0–100, higher = healthier) ---
    health_score = round((1 - failure_prob) * 100, 2)

    # --- Risk Category ---
    if health_score > 80:
        risk = "Low Risk"
        maintenance_days = 30
    elif health_score > 60:
        risk = "Medium Risk"
        maintenance_days = 14
    elif health_score > 40:
        risk = "High Risk"
        maintenance_days = 7
    else:
        risk = "Critical Risk"
        maintenance_days = 1

    # --- Extract sensor values for rule-based recommendations ---
    # Build a dict keyed by feature name for safe lookup
    data_dict = dict(zip(feature_columns, input_data))

    tool_wear  = data_dict.get('Tool wear min', 0)
    torque     = data_dict.get('Torque Nm', 0)
    proc_temp  = data_dict.get('Process temperature K', 0)
    air_temp   = data_dict.get('Air temperature K', 0)
    rpm        = data_dict.get('Rotational speed rpm', 0)

    recommendations = []

    if tool_wear > 200:
        recommendations.append("⚠️ Tool wear is high — replace tool within 2 days")
    elif tool_wear > 150:
        recommendations.append("🔧 Tool wear approaching limit — schedule replacement soon")

    if torque > 80:
        recommendations.append("⚠️ High torque detected — inspect mechanical components")

    # Process temperature thresholds in Kelvin (normal range ~305–313 K)
    if proc_temp > 315:
        recommendations.append("🌡️ Process temperature elevated — check cooling system")

    if air_temp > 305:
        recommendations.append("🌡️ Air temperature elevated — check ambient conditions")

    if rpm < 1200:
        recommendations.append("⚙️ Low rotational speed — check motor and drive belt")
    elif rpm > 2800:
        recommendations.append("⚙️ High rotational speed — check for vibration/imbalance")

    # Sub-failure type flags
    if data_dict.get('TWF', 0) == 1:
        recommendations.append("🔴 Tool Wear Failure flag active — immediate tool replacement required")
    if data_dict.get('HDF', 0) == 1:
        recommendations.append("🔴 Heat Dissipation Failure flag active — inspect cooling immediately")
    if data_dict.get('PWF', 0) == 1:
        recommendations.append("🔴 Power Failure flag active — check power supply and motor")
    if data_dict.get('OSF', 0) == 1:
        recommendations.append("🔴 Overstrain Failure flag active — reduce load immediately")
    if data_dict.get('RNF', 0) == 1:
        recommendations.append("🔴 Random Failure flag active — full diagnostic inspection needed")

    if not recommendations:
        recommendations.append("✅ All parameters normal — no immediate maintenance required")

    return {
        "Prediction": "Failure" if pred == 1 else "Healthy",
        "Failure Probability": round(failure_prob * 100, 2),
        "Health Score": health_score,
        "Risk": risk,
        "Maintenance Due (days)": maintenance_days,
        "Recommendations": recommendations,
    }

# ⚙️ Machine Failure Prediction & Maintenance Analytics Platform

A production-ready machine learning platform that predicts machine failures from real-time sensor data and provides actionable maintenance recommendations — built with **XGBoost** and **Streamlit**.

---

## 🚀 Live Demo
[live demo](https://machine-failure-prediction---maintenance-analytics-platform-it.streamlit.app/)
---

## 📁 Project Structure

```
Machine-Failure-Prediction-&-Maintenance-Analytics-Platform/
│
├── app/
│   └── streamlit_app.py        # Main Streamlit web application
│
├── data/
│   └── machine_failure_prediction_dataset.csv   # Raw dataset
│
├── models/
│   ├── xgboost_machine_failure_model.pkl        # Trained XGBoost model
│   ├── feature_columns.pkl                      # Feature column names
│   └── label_encoder.pkl                        # LabelEncoder for 'Type'
│
├── notebooks/
│   └── machine_failure_prediction_EDA.ipynb     # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data loading & preprocessing
│   ├── train_model.py          # Model training script
│   ├── predict.py              # Prediction & maintenance analytics logic
│   └── utils.py                # Helper utilities
│
├── .streamlit/
│   └── config.toml             # Streamlit theme & server config
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 Features

- **Failure Prediction** — Binary classification (Healthy / Failure) using XGBoost
- **Health Score** — 0–100 score derived from failure probability
- **Risk Categorization** — Low / Medium / High / Critical
- **Maintenance Timeline** — Estimated days until maintenance is required
- **Rule-based Recommendations** — Sensor-threshold alerts for operators
- **Data Analytics** — EDA charts, correlation heatmap, scatter plots

---

## 📊 Dataset

The dataset contains **10,000 synthetic records** of manufacturing machine sensor readings with labelled failure events.

| Column | Description |
|---|---|
| Type | Machine quality variant (L / M / H) |
| Air temperature [K] | Ambient air temperature in Kelvin |
| Process temperature [K] | Process temperature in Kelvin |
| Rotational speed [rpm] | Spindle rotational speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Cumulative tool wear time |
| Machine failure | Target label (0 = Healthy, 1 = Failure) |
| TWF / HDF / PWF / OSF / RNF | Sub-failure type flags |

---

## 🤖 Model

| Property | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Features | 11 (Type + 5 sensors + 5 sub-failure flags) |
| Target | Machine failure (binary) |
| Class imbalance | `scale_pos_weight` |
| Train/Test split | 80 / 20 (stratified) |

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/theamitrawat/Machine-Failure-Prediction-&-Maintenance-Analytics-Platform.git
cd Machine-Failure-Prediction-&-Maintenance-Analytics-Platform
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app/streamlit_app.py
```

---

## 🔁 Retrain the Model

To retrain from scratch with the existing dataset:

```bash
python src/train_model.py
```

This will overwrite `models/xgboost_machine_failure_model.pkl`, `models/feature_columns.pkl`, and `models/label_encoder.pkl`.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| XGBoost | ML model |
| scikit-learn | Preprocessing & metrics |
| Streamlit | Web UI |
| Plotly | Interactive charts |
| Pandas / NumPy | Data manipulation |
| Joblib | Model serialization |

---

## 👤 Author

**Amit Rawat**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/theamitrawat/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/theamitrawat)

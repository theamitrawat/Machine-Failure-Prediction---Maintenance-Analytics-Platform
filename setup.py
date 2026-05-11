from setuptools import setup, find_packages

setup(
    name="machine-failure-prediction",
    version="1.0.0",
    author="Amit Rawat",
    author_email="",
    description="Machine Failure Prediction & Maintenance Analytics Platform",
    url="https://github.com/theamitrawat/Machine-Failure-Prediction-&-Maintenance-Analytics-Platform",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas==2.3.3",
        "numpy==2.2.6",
        "scikit-learn==1.7.2",
        "xgboost==3.2.0",
        "matplotlib==3.10.3",
        "seaborn==0.13.2",
        "plotly==6.7.0",
        "streamlit==1.57.0",
        "joblib==1.5.3",
        "scipy==1.15.3",
    ],
)

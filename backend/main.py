from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

FORECAST_MODEL_PATH = MODEL_DIR / "random_forest.pkl"
FAULT_MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"

PREDICTION_DATA_PATH = (
    BASE_DIR.parent
    / "data"
    / "processed"
    / "test_predictions.csv"
)

FEATURE_IMPORTANCE_PATH = (
    BASE_DIR.parent
    / "data"
    / "processed"
    / "feature_importance.csv"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading Random Forest model...")
forecast_model = joblib.load(FORECAST_MODEL_PATH)

print("Loading Isolation Forest model...")
fault_model = joblib.load(FAULT_MODEL_PATH)

print("Models loaded successfully!")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SolarSense API",
    description="Solar Power Forecasting and Anomaly Detection API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# INPUT MODEL
# ============================================================

class SolarInput(BaseModel):

    ambient_temperature: float
    module_temperature: float
    irradiance: float

    hour: int
    minute: int
    day: int
    month: int
    day_of_week: int

    is_day: int

    lag_power: float
    rolling_power: float


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "SolarSense API is running",
        "forecast_model": "Random Forest",
        "fault_model": "Isolation Forest"
    }


# ============================================================
# SOLAR POWER PREDICTION
# ============================================================

@app.post("/api/predict")
def predict_power(data: SolarInput):

    input_data = pd.DataFrame([{

        "AMBIENT_TEMPERATURE":
            data.ambient_temperature,

        "MODULE_TEMPERATURE":
            data.module_temperature,

        "IRRADIATION":
            data.irradiance,

        "hour":
            data.hour,

        "minute":
            data.minute,

        "day":
            data.day,

        "month":
            data.month,

        "day_of_week":
            data.day_of_week,

        "is_day":
            data.is_day,

        "lag_power":
            data.lag_power,

        "rolling_power":
            data.rolling_power
    }])


    prediction = forecast_model.predict(
        input_data
    )[0]


    prediction = max(
        0,
        float(prediction)
    )


    return {

        "predicted_power":
            round(prediction, 2),

        "unit":
            "kW",

        "model":
            "Random Forest"
    }


# ============================================================
# ANOMALY DETECTION
# ============================================================

@app.post("/api/detect-fault")
def detect_fault(data: SolarInput):

    fault_input = pd.DataFrame([{

        "AC_POWER":
            data.lag_power,

        "AMBIENT_TEMPERATURE":
            data.ambient_temperature,

        "MODULE_TEMPERATURE":
            data.module_temperature,

        "IRRADIATION":
            data.irradiance
    }])


    prediction = fault_model.predict(
        fault_input
    )[0]


    anomaly_score = fault_model.decision_function(
        fault_input
    )[0]


    if prediction == -1:

        status = "⚠️ POSSIBLE ANOMALY"

    else:

        status = "🟢 NORMAL"


    return {

        "status":
            status,

        "anomaly_score":
            round(
                float(anomaly_score),
                4
            ),

        "model":
            "Isolation Forest"
    }


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.get("/api/model-performance")
def model_performance():

    return {

        "models": [

            {
                "name":
                    "Linear Regression",

                "mae":
                    23.9867,

                "rmse":
                    53.0029,

                "r2":
                    0.9820
            },

            {
                "name":
                    "Random Forest",

                "mae":
                    12.8316,

                "rmse":
                    37.7222,

                "r2":
                    0.9909
            },

            {
                "name":
                    "XGBoost",

                "mae":
                    14.7514,

                "rmse":
                    39.2492,

                "r2":
                    0.9901
            }
        ],

        "best_model":
            "Random Forest"
    }


# ============================================================
# ANOMALY SUMMARY
# ============================================================

@app.get("/api/anomaly-summary")
def anomaly_summary():

    return {

        "total_observations":
            68686,

        "normal_observations":
            67312,

        "anomalies":
            1374,

        "anomaly_percentage":
            2.00,

        "model":
            "Isolation Forest"
    }


# ============================================================
# ACTUAL VS PREDICTED CHART
# ============================================================

@app.get("/api/prediction-chart")
def prediction_chart():

    if not PREDICTION_DATA_PATH.exists():

        return {

            "error":
                "Prediction data not found. "
                "Run ml/08_generate_predictions.py first."
        }


    data = pd.read_csv(
        PREDICTION_DATA_PATH
    )


    sample_size = min(
        100,
        len(data)
    )


    chart_data = data.iloc[
        :sample_size
    ]


    return {

        "dates":
            chart_data[
                "DATE_TIME"
            ].tolist(),

        "actual":
            chart_data[
                "actual_power"
            ].round(2).tolist(),

        "predicted":
            chart_data[
                "predicted_power"
            ].round(2).tolist()
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

@app.get("/api/feature-importance")
def feature_importance():

    if not FEATURE_IMPORTANCE_PATH.exists():

        return {

            "error":
                "Feature importance data not found. "
                "Run ml/09_feature_importance.py first."
        }


    data = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )


    return {

        "features":
            data["feature"].tolist(),

        "importance":
            data["importance"].round(4).tolist()
    }
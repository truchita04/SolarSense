import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import IsolationForest
import joblib


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "solar_features.csv"
)

model_dir = BASE_DIR / "backend" / "models"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "isolation_forest.pkl"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading feature dataset...")

data = pd.read_csv(input_path)

print("Dataset shape:", data.shape)


# --------------------------------------------------
# Select anomaly detection features
# --------------------------------------------------

features = [
    "AC_POWER",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION"
]

X = data[features]


# --------------------------------------------------
# Train Isolation Forest
# --------------------------------------------------

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

model.fit(X)


# --------------------------------------------------
# Detect anomalies
# --------------------------------------------------

predictions = model.predict(X)

data["anomaly"] = predictions


# Isolation Forest:
#  1  = normal
# -1  = anomaly

data["fault_status"] = data["anomaly"].map({
    1: "NORMAL",
    -1: "ANOMALY"
})


# --------------------------------------------------
# Count results
# --------------------------------------------------

normal_count = (data["anomaly"] == 1).sum()

anomaly_count = (data["anomaly"] == -1).sum()

total = len(data)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n======================================")
print("FAULT / ANOMALY DETECTION RESULTS")
print("======================================")

print("Total observations :", total)

print("Normal observations:", normal_count)

print("Anomalies detected  :", anomaly_count)

print(
    f"Anomaly percentage  : "
    f"{(anomaly_count / total) * 100:.2f}%"
)


# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(model, model_path)

print("\nModel saved at:")

print(model_path)
import pandas as pd
import numpy as np

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "processed" / "solar_features.csv"

model_dir = BASE_DIR / "backend" / "models"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "xgboost.pkl"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading feature dataset...")

data = pd.read_csv(input_path)

print("Dataset shape:", data.shape)


# --------------------------------------------------
# Features
# --------------------------------------------------

features = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "hour",
    "minute",
    "day",
    "month",
    "day_of_week",
    "is_day",
    "lag_power",
    "rolling_power"
]

target = "AC_POWER"

X = data[features]
y = data[target]


# --------------------------------------------------
# Chronological train-test split
# --------------------------------------------------

split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# Train XGBoost
# --------------------------------------------------

print("\nTraining XGBoost...")

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n======================================")
print("XGBOOST RESULTS")
print("======================================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(model, model_path)

print("\nModel saved at:")
print(model_path)
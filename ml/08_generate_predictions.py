from pathlib import Path
import pandas as pd
import joblib


# ==============================
# PROJECT PATHS
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "solar_features.csv"
MODEL_PATH = BASE_DIR / "backend" / "models" / "random_forest.pkl"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "test_predictions.csv"


# ==============================
# LOAD DATA
# ==============================
print("Loading processed dataset...")

data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)


# ==============================
# PREPARE FEATURES
# ==============================
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


# ==============================
# TRAIN / TEST SPLIT
# ==============================
split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_data = data.iloc[split_index:].copy()


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# LOAD RANDOM FOREST MODEL
# ==============================
print("Loading Random Forest model...")

model = joblib.load(MODEL_PATH)


# ==============================
# MAKE PREDICTIONS
# ==============================
print("Generating predictions...")

predictions = model.predict(X_test)

predictions = predictions.clip(min=0)


# ==============================
# CREATE OUTPUT DATASET
# ==============================
result = pd.DataFrame({
    "DATE_TIME": test_data["DATE_TIME"].values,
    "SOURCE_KEY": test_data["SOURCE_KEY"].values,
    "actual_power": y_test.values,
    "predicted_power": predictions
})


# Calculate prediction error
result["error"] = (
    result["actual_power"] - result["predicted_power"]
).abs()


# ==============================
# SAVE RESULTS
# ==============================
result.to_csv(OUTPUT_PATH, index=False)


print("\nPrediction file created successfully!")
print("Saved to:", OUTPUT_PATH)

print("\nFirst 10 predictions:")
print(result.head(10))

print("\nPrediction summary:")
print(result[["actual_power", "predicted_power", "error"]].describe())
from pathlib import Path
import pandas as pd
import joblib


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "backend"
    / "models"
    / "random_forest.pkl"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_importance.csv"
)


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print("Loading Random Forest model...")

model = joblib.load(MODEL_PATH)


# ============================================================
# FEATURE NAMES
# ============================================================

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


# ============================================================
# GET FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_


# ============================================================
# CREATE DATAFRAME
# ============================================================

result = pd.DataFrame({
    "feature": features,
    "importance": importance
})


# ============================================================
# SORT
# ============================================================

result = result.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\nFeature Importance:")
print("=" * 50)

for index, row in result.iterrows():

    print(
        f"{index + 1:2}. "
        f"{row['feature']:<25} "
        f"{row['importance']:.4f}"
    )


print("\nFeature importance file created successfully!")

print(
    "Saved to:",
    OUTPUT_PATH
)
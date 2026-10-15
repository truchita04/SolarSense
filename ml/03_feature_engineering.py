import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "processed" / "solar_processed.csv"
output_path = BASE_DIR / "data" / "processed" / "solar_features.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading processed dataset...")

data = pd.read_csv(input_path)

print("Original shape:", data.shape)


# --------------------------------------------------
# Convert timestamp
# --------------------------------------------------

data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

data = data.sort_values(
    ["SOURCE_KEY", "DATE_TIME"]
).reset_index(drop=True)


# --------------------------------------------------
# Time features
# --------------------------------------------------

data["hour"] = data["DATE_TIME"].dt.hour

data["minute"] = data["DATE_TIME"].dt.minute

data["day"] = data["DATE_TIME"].dt.day

data["month"] = data["DATE_TIME"].dt.month

data["day_of_week"] = data["DATE_TIME"].dt.dayofweek


# --------------------------------------------------
# Day / night indicator
# --------------------------------------------------

data["is_day"] = (
    data["IRRADIATION"] > 0
).astype(int)


# --------------------------------------------------
# Inverter-specific lag
# --------------------------------------------------

data["lag_power"] = (
    data.groupby("SOURCE_KEY")["AC_POWER"]
    .shift(1)
)


# --------------------------------------------------
# Inverter-specific rolling average
# --------------------------------------------------

data["rolling_power"] = (
    data.groupby("SOURCE_KEY")["AC_POWER"]
    .transform(
        lambda x: x.shift(1).rolling(4).mean()
    )
)


# --------------------------------------------------
# Remove missing lag/rolling rows
# --------------------------------------------------

data = data.dropna().reset_index(drop=True)


# --------------------------------------------------
# Select final columns
# --------------------------------------------------

data = data[
    [
        "DATE_TIME",
        "PLANT_ID",
        "SOURCE_KEY",
        "AC_POWER",
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
]


# --------------------------------------------------
# Save
# --------------------------------------------------

data.to_csv(output_path, index=False)


print("\n======================================")
print("INVERTER-SPECIFIC FEATURE ENGINEERING")
print("======================================")

print("\nFinal shape:", data.shape)

print("\nFinal columns:")
print(data.columns.tolist())

print("\nTarget:")
print("AC_POWER")

print("\nSaved at:")
print(output_path)
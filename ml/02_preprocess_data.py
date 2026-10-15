import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

generation_path = BASE_DIR / "data" / "raw" / "Plant_1_Generation_Data.csv"
weather_path = BASE_DIR / "data" / "raw" / "Plant_1_Weather_Sensor_Data.csv"

output_dir = BASE_DIR / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "solar_processed.csv"


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("Loading datasets...")

generation = pd.read_csv(generation_path)
weather = pd.read_csv(weather_path)

print("Generation shape:", generation.shape)
print("Weather shape:", weather.shape)


# --------------------------------------------------
# Convert DATE_TIME to datetime
# --------------------------------------------------

print("\nConverting DATE_TIME...")

generation["DATE_TIME"] = pd.to_datetime(
    generation["DATE_TIME"],
    dayfirst=True
)

weather["DATE_TIME"] = pd.to_datetime(
    weather["DATE_TIME"]
)


# --------------------------------------------------
# Merge generation and weather data
# --------------------------------------------------

print("\nMerging datasets...")

data = pd.merge(
    generation,
    weather,
    on=["DATE_TIME", "PLANT_ID"],
    how="inner"
)


# --------------------------------------------------
# Sort data by time
# --------------------------------------------------

data = data.sort_values("DATE_TIME").reset_index(drop=True)


# --------------------------------------------------
# Remove unnecessary columns
# --------------------------------------------------

# Keep the generation inverter/source key.
# Weather SOURCE_KEY is not required because weather
# measurements are associated with the plant.

data = data.rename(
    columns={"SOURCE_KEY_x": "SOURCE_KEY"}
)

data = data.drop(
    columns=["SOURCE_KEY_y"],
    errors="ignore"
)


# --------------------------------------------------
# Select important columns
# --------------------------------------------------
data = data[
    [
        "DATE_TIME",
        "PLANT_ID",
        "SOURCE_KEY",
        "DC_POWER",
        "AC_POWER",
        "DAILY_YIELD",
        "TOTAL_YIELD",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]
]


# --------------------------------------------------
# Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(data.isnull().sum())


# --------------------------------------------------
# Remove duplicate records
# --------------------------------------------------

before = len(data)

data = data.drop_duplicates()

after = len(data)

print("\nDuplicates removed:", before - after)


# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------

data.to_csv(output_path, index=False)

print("\n======================================")
print("PREPROCESSING COMPLETED")
print("======================================")

print("Final shape:", data.shape)

print("\nFinal columns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nProcessed file saved at:")
print(output_path)
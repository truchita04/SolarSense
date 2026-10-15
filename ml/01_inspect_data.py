import pandas as pd
import os


# ---------------------------------------
# File paths
# ---------------------------------------

generation_path = "data/raw/Plant_1_Generation_Data.csv"
weather_path = "data/raw/Plant_1_Weather_Sensor_Data.csv"


# ---------------------------------------
# Load datasets
# ---------------------------------------

generation = pd.read_csv(generation_path)
weather = pd.read_csv(weather_path)


# ---------------------------------------
# Basic information
# ---------------------------------------

print("\n========== GENERATION DATA ==========\n")

print(generation.head())

print("\nShape:")
print(generation.shape)

print("\nColumns:")
print(generation.columns.tolist())

print("\nData types:")
print(generation.dtypes)

print("\nMissing values:")
print(generation.isnull().sum())


print("\n========== WEATHER DATA ==========\n")

print(weather.head())

print("\nShape:")
print(weather.shape)

print("\nColumns:")
print(weather.columns.tolist())

print("\nData types:")
print(weather.dtypes)

print("\nMissing values:")
print(weather.isnull().sum())


# ---------------------------------------
# Duplicate check
# ---------------------------------------

print("\n========== DUPLICATES ==========\n")

print(
    "Generation duplicates:",
    generation.duplicated().sum()
)

print(
    "Weather duplicates:",
    weather.duplicated().sum()
)
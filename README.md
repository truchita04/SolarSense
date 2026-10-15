# SOLAR-SENSE
## Solar Power Prediction and Fault Detection Using Machine Learning

---

## 1. Project Overview

SolarSense is a machine learning based solar power monitoring and prediction system.

The system uses historical solar plant data to:

1. Predict solar power generation.
2. Detect potential abnormal operating conditions.
3. Compare different machine learning models.
4. Identify the most important factors affecting solar power generation.
5. Provide predictions through a FastAPI backend.
6. Display results through a web-based frontend dashboard.

The project combines Machine Learning, FastAPI, and a web-based user interface into a complete end-to-end application.

---

## 2. Problem Statement

Solar power generation depends on several environmental and operational factors such as solar irradiation, ambient temperature, module temperature, time of day, and previous power generation.

Because solar generation is variable, it is difficult to accurately estimate the amount of power that will be generated at a particular time.

In addition, abnormal operating conditions may reduce the performance of a solar plant.

Therefore, there is a need for an intelligent system that can:

- Predict solar power generation.
- Identify potentially abnormal operating conditions.
- Analyze the factors influencing solar power generation.
- Provide the results through an easy-to-use interface.

SolarSense addresses this problem using supervised and unsupervised machine learning techniques.

---

## 3. Objectives

The main objectives of SolarSense are:

- To preprocess historical solar power generation data.
- To perform feature engineering on solar plant data.
- To train multiple machine learning regression models.
- To compare model performance using MAE, RMSE, and R².
- To select the best-performing model for solar power prediction.
- To detect potential anomalies using Isolation Forest.
- To analyze feature importance using Random Forest.
- To develop a FastAPI backend for model deployment.
- To create a web dashboard for user interaction.
- To provide an end-to-end machine learning application.

---

## 4. Dataset

The project uses historical solar power plant data containing information related to:

- Solar power generation
- Ambient temperature
- Module temperature
- Solar irradiation
- Date and time
- Inverter/source information

The data is processed before being supplied to the machine learning models.

---

## 5. Data Preprocessing

The preprocessing stage includes:

- Loading the raw dataset.
- Converting date and time information into datetime format.
- Sorting observations by source/inverter and timestamp.
- Handling missing values.
- Removing unnecessary columns.
- Preparing the data for machine learning.

---

## 6. Feature Engineering

Additional features were created from the original dataset.

The final prediction features include:

- Ambient Temperature
- Module Temperature
- Irradiation
- Hour
- Minute
- Day
- Month
- Day of Week
- Is Day
- Previous Power
- Previous Average Power

Lag and rolling features are calculated separately for each inverter/source to preserve the temporal relationship within each source.

---

## 7. Machine Learning Models

Three supervised machine learning regression models were trained:

### Linear Regression

Linear Regression provides a simple baseline model for predicting solar power.

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

It achieved the best performance among the tested regression models and was selected as the primary prediction model.

### XGBoost

XGBoost is a gradient boosting algorithm that builds an ensemble of decision trees sequentially.

---

## 8. Model Evaluation

The models were evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

Lower MAE indicates better performance.

### Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction error.

Lower RMSE indicates better performance.

### R² Score

Measures how well the model explains the variation in the target variable.

Higher R² indicates better performance.

### Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 23.9867 | 53.0029 | 0.9820 |
| Random Forest | 12.8316 | 37.7222 | 0.9909 |
| XGBoost | 14.7514 | 39.2492 | 0.9901 |

Random Forest achieved the best overall performance.

Therefore, Random Forest is used as the primary solar power prediction model in the deployed application.

---

## 9. Anomaly Detection

SolarSense uses Isolation Forest for unsupervised anomaly detection.

The model analyzes:

- AC Power
- Ambient Temperature
- Module Temperature
- Irradiation

The model identifies observations that differ significantly from normal operating patterns.

### Results

Total observations:

68686

Normal observations:

67312

Potential anomalies:

1374

Anomaly rate:

2.00%

The detected observations should be interpreted as potential abnormal operating conditions rather than confirmed physical equipment faults because the dataset does not contain ground-truth fault labels.

---

## 10. Feature Importance

Random Forest feature importance is used to identify which input variables contribute most strongly to the model's predictions.

The feature importance results are displayed in the SolarSense dashboard.

This provides model interpretability and helps understand the factors influencing solar power generation.

---

## 11. System Architecture

The system consists of three major layers.

### Machine Learning Layer

Responsible for:

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Prediction generation
- Anomaly detection
- Feature importance analysis

### Backend Layer

FastAPI provides REST API endpoints for:

- Solar power prediction
- Anomaly detection
- Model performance
- Prediction chart data
- Feature importance

### Frontend Layer

The web dashboard provides:

- User input fields
- Solar power prediction
- Anomaly detection
- Actual vs predicted chart
- Feature importance chart
- Model comparison
- Anomaly statistics

---

## 12. Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- XGBoost
- Pandas
- NumPy

### Backend

- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript
- Chart.js

### Model Serialization

- Joblib

---

## 13. Project Structure

```text
SOLAR-SENSE/
│
├── backend/
│   ├── main.py
│   └── models/
│       ├── linear_regression.pkl
│       ├── random_forest.pkl
│       ├── xgboost.pkl
│       └── isolation_forest.pkl
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── ml/
│   ├── 01_inspect_data.py
│   ├── 02_preprocess_data.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_linear_regression.py
│   ├── 05_train_random_forest.py
│   ├── 06_train_xgboost.py
│   ├── 07_train_fault_detection.py
│   ├── 08_generate_predictions.py
│   └── 09_feature_importance.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── solar_features.csv
│       ├── test_predictions.csv
│       └── feature_importance.csv
│
└── README.md
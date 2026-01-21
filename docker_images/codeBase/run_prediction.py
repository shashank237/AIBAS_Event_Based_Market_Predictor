import pandas as pd
import tensorflow as tf
import statsmodels.api as sm
import joblib
import os

# Paths inside containers
ACTIVATION_PATH = "/tmp/activation/activation_data.csv"
ANN_MODEL_PATH  = "/tmp/knowledge/models/currentAiSolution.h5"
OLS_MODEL_PATH  = "/tmp/knowledge/models/currentOlsSolution.pkl"
SCALER_PATH     = "/tmp/knowledge/models/scaler.pkl"
OUTPUT_PATH     = "/tmp/output/predictions.csv"

# Load activation data
activation_df = pd.read_csv(ACTIVATION_PATH)

# Feature engineering (must match training)
activation_df["Date"] = pd.to_datetime(activation_df["Date"])
activation_df["Year"]  = activation_df["Date"].dt.year
activation_df["Month"] = activation_df["Date"].dt.month
activation_df["Day"]   = activation_df["Date"].dt.day

X = activation_df.drop(columns=["Date", "Close_Price"], errors="ignore")

# Load models
ann = tf.keras.models.load_model(ANN_MODEL_PATH)
ols = joblib.load(OLS_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ANN prediction
X_scaled = scaler.transform(X)
ann_pred = ann.predict(X_scaled).flatten()[0]

# OLS prediction
X_ols = sm.add_constant(X, has_constant="add")
ols_pred = ols.predict(X_ols)[0]

# Save output
os.makedirs("/tmp/output", exist_ok=True)

output_df = pd.DataFrame({
    "ANN_Prediction": [ann_pred],
    "OLS_Prediction": [ols_pred]
})

output_df.to_csv(OUTPUT_PATH, index=False)

print("=== Event-Based Market Prediction ===")
print(output_df)

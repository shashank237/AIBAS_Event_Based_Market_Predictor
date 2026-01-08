import pandas as pd
import numpy as np
import tensorflow as tf
import statsmodels.api as sm
import joblib
import os

# === Load activation data ===
activation_df = pd.read_csv("/activation/activation_data.csv")

# === Load models ===
ann = tf.keras.models.load_model("/knowledge/models/currentAiSolution.h5")
ols = joblib.load("/knowledge/models/currentOlsSolution.pkl")
scaler = joblib.load("/knowledge/models/scaler.pkl")

# === Feature engineering (must match training) ===
activation_df["Date"] = pd.to_datetime(activation_df["Date"])
activation_df["Year"] = activation_df["Date"].dt.year
activation_df["Month"] = activation_df["Date"].dt.month
activation_df["Day"] = activation_df["Date"].dt.day

X = activation_df.drop(columns=["Date", "Close_Price"])

# === ANN Prediction ===
X_scaled = scaler.transform(X)
ann_pred = ann.predict(X_scaled).flatten()[0]

# === OLS Prediction ===
X_ols = sm.add_constant(X)
ols_pred = ols.predict(X_ols)[0]

print("\n=== Activation Results ===")
print("ANN Prediction:", ann_pred)
print("OLS Prediction:", ols_pred)

# === Save output to shared volume ===
OUTPUT_DIR = "/tmp/activationBase/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_df = pd.DataFrame({
    "ANN_Prediction": [ann_pred],
    "OLS_Prediction": [ols_pred]
})

output_path = f"{OUTPUT_DIR}/predictions.csv"
output_df.to_csv(output_path, index=False)

print(f"\nResults written to {output_path}")

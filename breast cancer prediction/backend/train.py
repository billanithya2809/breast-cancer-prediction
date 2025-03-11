import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ========================== Step 1: Load & Clean Data ==========================

# Load dataset
df = pd.read_csv("backend/data.csv")

# Drop ID Column (if exists)
df.drop(columns=['id'], inplace=True, errors='ignore')

# Encode Diagnosis (M = 1 for Malignant, B = 0 for Benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Fill missing values with column mean
df.fillna(df.mean(), inplace=True)

# ========================== Step 2: Feature Selection ==========================

feature_cols = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
    "symmetry_mean", "fractal_dimension_mean", "radius_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst"
]

# Standardize numerical features
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# ========================== Step 3: Train ER Status & HER2 Status Models ==========================

# Simulating ER/HER2 labels (Use actual labels if available)
df["ER_Status_Label"] = np.random.choice([0, 1], size=len(df))
df["HER2_Status_Label"] = np.random.choice([0, 1], size=len(df))

# Train classifiers
er_model = RandomForestClassifier(n_estimators=100, random_state=42)
her2_model = RandomForestClassifier(n_estimators=100, random_state=42)
er_model.fit(df[feature_cols], df["ER_Status_Label"])
her2_model.fit(df[feature_cols], df["HER2_Status_Label"])

categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Assign predictions
df["ER_Status"] = er_model.predict(df[feature_cols]).astype(str)
df["HER2_Status"] = her2_model.predict(df[feature_cols]).astype(str)

# Convert categorical values
df["ER_Status"] = df["ER_Status"].map({"1": "Positive", "0": "Negative"})
df["HER2_Status"] = df["HER2_Status"].map({"1": "Positive", "0": "Negative"})

# ========================== Step 4: Train Tumor Size Prediction Model ==========================

X_size = df[["radius_mean", "perimeter_mean", "area_mean"]]
y_size = df["area_mean"] / 10  # Approximate tumor size

size_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
size_regressor.fit(X_size, y_size)

df["Invasive_Tumor_Size_mm"] = size_regressor.predict(X_size)

# Save updated dataset
df.to_csv("breast_cancer_extended.csv", index=False)

# Save models
joblib.dump(er_model, "er_status_model.pkl")
joblib.dump(her2_model, "her2_status_model.pkl")
joblib.dump(size_regressor, "tumor_size_model.pkl")

print("✅ Dataset updated with ER Status, HER2 Status, and Invasive Tumor Size!")

import os

# Define path to save models
MODEL_DIR = "backend/models/"
os.makedirs(MODEL_DIR, exist_ok=True)  # ✅ Ensure directory exists

# Save models
joblib.dump(er_model, os.path.join(MODEL_DIR, "er_status_model.pkl"))
joblib.dump(her2_model, os.path.join(MODEL_DIR, "her2_status_model.pkl"))
joblib.dump(size_regressor, os.path.join(MODEL_DIR, "tumor_size_model.pkl"))


print(f"✅ Models saved in {MODEL_DIR}")

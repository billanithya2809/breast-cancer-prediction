import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
# ========================== Step 1: Load Dataset ==========================
df = pd.read_csv("breast_cancer_extended.csv")

# Ensure categorical features are strings and avoid warnings
X = df.copy()

# ========================== Step 2: Feature Selection ==========================
selected_features = [
    "ER_Status", "HER2_Status", "Invasive_Tumor_Size_mm",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
    "symmetry_mean", "fractal_dimension_mean", "radius_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "concavity_worst", "concave points_worst"
]

# Select features and target variable
X = X[selected_features].copy()  # Ensure we're working on a copy
y = df["diagnosis"]

# Convert categorical features to strings and avoid SettingWithCopyWarning
X.loc[:, "ER_Status"] = X["ER_Status"].astype(str)
X.loc[:, "HER2_Status"] = X["HER2_Status"].astype(str)

# Define categorical and numerical columns
categorical_cols = ["ER_Status", "HER2_Status"]
numerical_cols = [col for col in selected_features if col not in categorical_cols]

# ========================== Step 3: Define Preprocessing Pipeline ==========================
categorical_transformer = OneHotEncoder(handle_unknown="ignore", dtype=np.float64, sparse_output=False)
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# ========================== Step 4: Define Model Pipeline ==========================
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
])

# ========================== Step 5: Split Data and Train Model ==========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🔍 Training SVM model...")
pipeline.fit(X_train, y_train)

# ========================== Step 6: Save Model and Features ==========================
MODEL_DIR = "backend/models/"
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure directory exists

joblib.dump(pipeline, os.path.join(MODEL_DIR, "breast_cancer_svm_model.pkl"))
joblib.dump(selected_features, os.path.join(MODEL_DIR, "selected_features.pkl"))

print("✅ SVM Model trained successfully!")
print(f"✅ Models saved in {MODEL_DIR}")

"""
train_model.py

Run this in Colab or locally (with Python 3.10 packages) to create churn_pipeline.pkl.
"""

# Install exact versions if in Colab (uncomment)
# !pip install pandas==1.5.3 numpy==1.23.5 scikit-learn==1.2.2 joblib==1.2.0

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Replace 'WA_Fn-UseC_-Telco-Customer-Churn (1).csv' with your file name if different
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

# Basic cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
df["TotalCharges"].fillna(0, inplace=True)

# Map target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop ID if present
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify numeric and categorical cols
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
], remainder="drop")

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

print("Train accuracy:", pipeline.score(X_train, y_train))
print("Test accuracy:", pipeline.score(X_test, y_test))

# Save model (joblib)
joblib.dump(pipeline, "churn_pipeline.pkl")
print("churn_pipeline.pkl saved.")

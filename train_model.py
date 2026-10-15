
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load dataset retrieved from Azure Blob Storage
df = pd.read_csv("/tmp/ai4i2020.csv")

# Select predictors (avoid target leakage)
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = pd.get_dummies(df[features], columns=["Type"])
y = df["Machine failure"]

# Stratified split preserves class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train a baseline Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

print("\nFeature importance:")
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance)
from pathlib import Path
import joblib

Path("models").mkdir(exist_ok=True)

joblib.dump(
    {"model": model, "features": X.columns.tolist()},
    "models/random_forest.joblib"
)

print("Model saved successfully!")

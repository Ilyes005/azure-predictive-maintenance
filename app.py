
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Predictive Maintenance API")

# Load the model you trained earlier
bundle = joblib.load(
    Path(__file__).parent / "models" / "random_forest.joblib"
)
model = bundle["model"]
features = bundle["features"]


class MachineInput(BaseModel):
    machine_type: str = Field(pattern="^[LMH]$")
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running"}


@app.post("/predict")
def predict(machine: MachineInput):
    row = pd.DataFrame([{
        "Type": machine.machine_type,
        "Air temperature [K]": machine.air_temperature,
        "Process temperature [K]": machine.process_temperature,
        "Rotational speed [rpm]": machine.rotational_speed,
        "Torque [Nm]": machine.torque,
        "Tool wear [min]": machine.tool_wear
    }])

    # Apply the same encoding as during training
    row = pd.get_dummies(row, columns=["Type"])
    row = row.reindex(columns=features, fill_value=0)

    probability = float(model.predict_proba(row)[0, 1])

    return {
        "failure_probability": round(probability, 4),
        "predicted_failure": int(probability >= 0.5)
    }

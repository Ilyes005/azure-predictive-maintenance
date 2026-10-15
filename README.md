# Azure Predictive Maintenance — Machine Learning API

An end-to-end predictive maintenance project combining Microsoft Azure Blob Storage, Python, machine learning, and FastAPI to predict industrial equipment failures from sensor measurements.

## Project Overview

This project demonstrates a cloud-based data science workflow, from storing and retrieving industrial sensor data in Microsoft Azure to training a machine learning model and serving predictions through a REST API.

**Technologies:** Python, pandas, scikit-learn, Random Forest, Microsoft Azure Blob Storage, Azure CLI, FastAPI, Git.

## Architecture

1. **Cloud Storage:** Upload the AI4I 2020 dataset to Azure Blob Storage.
2. **Data Retrieval:** Access the dataset using Azure CLI and process it with pandas.
3. **Data Preparation:** Select sensor features, encode categorical variables, and create stratified training and testing sets.
4. **Machine Learning:** Train a Random Forest classifier to predict equipment failure.
5. **Model Evaluation:** Evaluate precision, recall, F1-score, ROC-AUC, and feature importance.
6. **Prediction API:** Serve predictions using FastAPI and the serialized scikit-learn model.

The API was tested locally within Azure Cloud Shell. Persistent cloud deployment is a potential future extension.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

The dataset contains 10,000 synthetic observations representing industrial machine operating conditions.

Input features include:
- Air and process temperatures
- Rotational speed
- Torque
- Tool wear
- Product quality category

**Target:** Machine failure (binary classification).

Failure-mode indicators were excluded from model inputs to avoid target leakage.

## Model Performance

The Random Forest classifier was evaluated on a stratified 20% test set.

| Metric | Result |
|---|---|
| Accuracy | Approximately 98% |
| ROC-AUC | 0.9613 |
| Failure precision | 0.72 |
| Failure recall | 0.65 |
| Failure F1-score | 0.68 |

**Confusion matrix:**

| | Predicted healthy | Predicted failure |
|---|---:|---:|
| Actually healthy | 1,915 | 17 |
| Actually failed | 24 | 44 |

The model correctly identified 44 of 68 machine failures in the test set.

The strongest predictors according to Random Forest feature importance were rotational speed, torque, and tool wear.

## REST API

The FastAPI service exposes a `POST /predict` endpoint that accepts machine operating measurements and returns a predicted failure class and estimated failure probability.

Example request:

```json
{
  "machine_type": "M",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551,
  "torque": 42.8,
  "tool_wear": 0
}
```

Example response from the tested model:

```json
{
  "failure_probability": 0.0,
  "predicted_failure": 0
}
```

## Running Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the model after making the dataset available at the path specified in `train_model.py`:

```bash
python train_model.py
```

Start the prediction API:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

## Limitations and Future Improvements

- The dataset is synthetic and does not represent validated real-world industrial performance.
- The reported metrics come from a single stratified train/test split.
- Failure recall could be improved through threshold tuning and model optimization.
- Cloud-hosted API deployment, automated testing, and CI/CD are planned improvements.

## Author

Ilyes Ihidheb

Statistics and Data Science — Ludwig Maximilian University of Munich

[GitHub Profile](https://github.com/Ilyes005)

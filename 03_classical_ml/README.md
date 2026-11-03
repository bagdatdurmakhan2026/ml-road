# 03 - Classical Machine Learning & Inference Service

This module contains custom implementations of classical ML algorithms (Logistic Regression, Linear Regression), training scripts, and a production-ready FastAPI microservice packaged with Docker.

## 📂 Contents
* `CustomLogisticRegression.py` — Custom implementation of Logistic Regression from scratch.
* `linear_regression.py` — Linear regression logic.
* `metrics.py` — Evaluation metrics.
* `train.py` — Training script that serializes the model into `model.pkl`.
* `app.py` — FastAPI application providing the `/predict` endpoint.
* `Dockerfile` — Configuration for containerizing the service.
* `requirements.txt` — Project dependencies for the module.

---

## 🚀 How to Run

### Option 1: Run Locally
1. Install dependencies, train the model, and run the API:
```bash
pip install -r requirements.txt
python train.py
python -m uvicorn app:app --reload 
```

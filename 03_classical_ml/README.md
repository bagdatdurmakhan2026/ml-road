# 🚀 End-to-End ML & MLOps Pipeline

A complete machine learning pipeline: from implementing custom models and training, to weight serialization and web service deployment.

## 🛠 Directory Contents

- **Algorithms & Training:**
  - **`linear_regression.py`** — implementation of linear regression.
  - **`CustomLogisticRegression.py`** — custom logistic regression algorithm.
  - **`metrics.py`** — calculation of model quality metrics.
  - **`train.py`** — model training script and weight serialization.

- **Infrastructure & Deployment:**
  - **`model.pkl`** — saved weights / trained model artifact.
  - **`app.py`** — web application / API for model inference.
  - **`2.py`** — auxiliary experimental scripts.

## 🚀 Quick Start

1. Train the model and save the artifact (`model.pkl`):
   ```bash
   python train.py
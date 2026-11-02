# 🚀 06_projects: Spam Classifier & API Service

An end-to-end practical project implementing a custom text classification model (Spam / Normal) from scratch, integrated into a production-grade REST API microservice.

## 🛠 Directory Structure

- **`spam_detector.py`**: Model training script built on pure NumPy.
  - Text vectorization using **Bag-of-Words (BoW)** without external ML frameworks.
  - Fully connected linear layer for weight prediction.
  - Model training powered by **Gradient Descent** (MSE Loss).

- **`spam_api.py`**: FastAPI microservice for real-time message inference.
  - Accepts JSON payloads with message text (`POST /predict`).
  - Returns prediction scores and final verdicts (🚨 SPAM / ✅ Normal).

## 🚀 Quick Start

1. Ensure dependencies are installed:
   ```bash
   pip install numpy fastapi uvicorn pydantic
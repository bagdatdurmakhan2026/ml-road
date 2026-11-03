# 🚀 Machine Learning & MLOps Roadmap

A hands-on repository dedicated to building Machine Learning models, classical algorithms, and production-grade MLOps pipelines from scratch.

---

## 🛠 Tech Stack

* **Languages & Libraries:** Python 3.14+, NumPy, Pandas, Pydantic
* **Server & API:** FastAPI, Uvicorn, Pickle (Serialization)
* **Tools:** Git, Swagger UI

---

## 📁 Project Structure

```text
.
├── 01_math/              # Mathematical foundations of ML (Linear Algebra, Calculus)
├── 02_numpy_pandas/      # Data manipulation, vectorization, preprocessing
├── 03_classical_ml/      # Custom ML algorithms & model inference
│   ├── app.py            # FastAPI service for predictions (/predict)
│   ├── train.py          # Training and model serialization script
│   └── model.pkl         # Serialized model artifact
├── 04_deep_learning/     # Deep learning & neural networks from scratch
├── 05_nlp/               # Natural Language Processing (NLP) foundations
└── 06_projects/          # End-to-end practical projects & microservices
```

---

## 🚀 Quick Start

1. **Clone repository & enter project folder:**
   ```bash
   git clone https://github.com/bagdatdurmakhan2026/ml-road.git
   cd 03_classical_ml
   ```

2. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn numpy pydantic
   ```

3. **Train model & generate artifact:**
   ```bash
   python train.py
   ```

4. **Run FastAPI application:**
   ```bash
   python -m uvicorn app:app --reload
   ```

5. **Test API:**
   Open Swagger UI at `http://127.0.0.1:8000/docs` to test endpoints.

---

## 📡 API Endpoints

### `POST /predict`
Predicts target class based on feature vector.

* **Request Body:**
  ```json
  {
    "features": [1.5]
  }
  ```

* **Response (200 OK):**
  ```json
  {
    "prediction": 1,
    "status": "success"
  }
  ```
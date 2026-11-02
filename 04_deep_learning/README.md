# 🚀 Neural Network & API Service

A module implementing a baseline neural network architecture from scratch using NumPy, integrated with a FastAPI web service.

## 🛠 Directory Structure

- **`neural_network.py`**:
  - Implementation of the neural network class featuring Forward Propagation.
  - Initialization of random weights and bias parameters.
  - Making predictions based on inputs without relying on external ML frameworks.

- **`api.py`**:
  - FastAPI microservice exposing neural network inference over HTTP.
  - Request data validation using **Pydantic**.
  - Interactive Swagger documentation for testing model predictions.

## 🚀 How to Run

1. Test the neural network script:
   ```bash
   python neural_network.py
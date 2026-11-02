from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Инициализируем приложение FastAPI
app = FastAPI(title="Custom Neural Network API", version="1.0")

# Описываем структуру данных, которую ждем от клиента (JSON)
class PredictionRequest(BaseModel):
    features: list[float]  # Например: [1.0, 2.0, 3.0]

# Создаем класс нашей сети (тот самый, который мы уже обучили)
class SimpleNeuralNetwork:
    def __init__(self):
        # Веса те же, можно будет потом вынести в файл, 
        # но для старта инициализируем структуру
        self.W1 = np.random.randn(3, 4) * 0.01
        self.b1 = np.zeros((1, 4))
        self.W2 = np.random.randn(4, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

# Создаем экземпляр модели при запуске сервера
model = SimpleNeuralNetwork()

@app.get("/")
def home():
    return {"message": "Neural Network API is running! Send POST to /predict"}

@app.post("/predict")
def predict(data: PredictionRequest):
    # Превращаем список из JSON в матрицу NumPy нужного формата
    X_input = np.array([data.features])
    
    # Делаем предсказание нашей сетью
    prediction = model.forward(X_input)
    
    # Возвращаем результат клиенту
    return {
        "input_features": data.features,
        "prediction": prediction.tolist()
    }
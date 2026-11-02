from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Инициализируем приложение
app = FastAPI(title="ML Linear Regression API")

# Загружаем сохраненную модель при старте сервера
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Описываем структуру входных данных с помощью Pydantic (для валидации)
class FeaturesInput(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: FeaturesInput):
    # Превращаем входящий список в numpy-массив нужной формы (1 строка, N признаков)
    X = np.array(data.features).reshape(1, -1)
    
    # Делаем предсказание нашей моделью
    prediction = model.predict(X)
    
    return {
        "prediction": float(prediction.item()),  # .item() безопасно достает скаляр из любого массива numpy
        "status": "success"
    }
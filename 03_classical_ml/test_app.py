from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict():
    # Отправляем тестовый POST-запрос с фичами
    response = client.post("/predict", json={"features": [1.5]})
    
    # Проверяем, что запрос прошел успешно (статус 200)
    assert response.status_code == 200
    
    # Проверяем структуру ответа
    data = response.json()
    assert "prediction" in data
    assert data["status"] == "success"
    assert isinstance(data["prediction"], float)
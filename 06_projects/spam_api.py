from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# 1. Обучаем нашу модель прямо при старте сервера (на том же мини-датасете)
corpus = [
    "free money click here",
    "hello how are you",
    "win a million dollars now",
    "lets grab lunch tomorrow"
]
labels = np.array([[1], [0], [1], [0]])

words = sorted(list(set(" ".join(corpus).split())))
vectors = [[text.split().count(word) for word in words] for text in corpus]
X = np.array(vectors)

class SimpleSpamClassifier:
    def __init__(self, input_dim):
        np.random.seed(42)
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = 0.0

    def forward(self, X):
        return np.dot(X, self.W) + self.b

    def train(self, X, y, epochs=200, lr=0.05):
        for _ in range(epochs):
            preds = self.forward(X)
            error = preds - y
            dW = np.dot(X.T, error) / len(X)
            db = np.mean(error)
            self.W -= lr * dW
            self.b -= lr * db

# Инициализируем и обучаем модель
model = SimpleSpamClassifier(input_dim=len(words))
model.train(X, labels)

# 2. Поднимаем FastAPI
app = FastAPI(title="Spam Detector API", version="1.0")

# Описываем структуру входящего JSON-запроса через Pydantic
class MessageRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(payload: MessageRequest):
    # Превращаем входящий текст пользователя в вектор по нашему словарю
    text_words = payload.text.split()
    vector = np.array([[text_words.count(word) for word in words]])
    
    # Делаем предсказание моделью
    pred_score = model.forward(vector)[0][0]
    is_spam = bool(pred_score > 0.5)
    
    return {
        "text": payload.text,
        "score": float(pred_score),
        "is_spam": is_spam,
        "verdict": "🚨 СПАМ" if is_spam else "✅ Норм"
    }
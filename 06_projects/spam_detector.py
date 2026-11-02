import numpy as np

# 1. Готовим данные: тексты и метки (0 — нормальное сообщение, 1 — спам)
corpus = [
    "free money click here",
    "hello how are you",
    "win a million dollars now",
    "lets grab lunch tomorrow"
]
labels = np.array([[1], [0], [1], [0]])  # Метки в виде колонки (4, 1)

# 2. Собираем уникальный словарь и матрицу признаков (X)
words = sorted(list(set(" ".join(corpus).split())))
vectors = [[text.split().count(word) for word in words] for text in corpus]
X = np.array(vectors)

# 3. Модель классификации с обучением
class SimpleSpamClassifier:
    def __init__(self, input_dim):
        np.random.seed(42)  # Чтобы веса инициализировались одинаково
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = 0.0

    def forward(self, X):
        return np.dot(X, self.W) + self.b

    def train(self, X, y, epochs=100, lr=0.1):
        for epoch in range(epochs):
            # Прямой проход (предсказание)
            preds = self.forward(X)
            
            # Считаем ошибку (MSE)
            loss = np.mean((preds - y) ** 2)
            
            # Градиенты (производные ошибки по весам)
            error = preds - y
            dW = np.dot(X.T, error) / len(X)
            db = np.mean(error)
            
            # Шаг градиентного спуска (обновляем веса)
            self.W -= lr * dW
            self.b -= lr * db

# 4. Создаем и обучаем модель
model = SimpleSpamClassifier(input_dim=X.shape[1])
model.train(X, labels, epochs=200, lr=0.05)

# 5. Проверяем результат после обучения
final_preds = model.forward(X)

print("\nРезультаты предсказаний ПОСЛЕ обучения:")
for text, pred, true_label in zip(corpus, final_preds, labels):
    status = "🚨 СПАМ" if pred[0] > 0.5 else "✅ Норм"
    print(f"  '{text}' -> Оценка: {pred[0]:.4f} | Исходная метка: {true_label[0]} | Вердикт: {status}")
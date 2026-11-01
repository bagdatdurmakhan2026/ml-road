import numpy as np
import matplotlib.pyplot as plt
class CustomLogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    def _sigmoid(self, z):
        # Ограничиваем z, чтобы экспонента не улетала в бесконечность (защита от переполнения)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        n_samples, n_features = X.shape    
        # 1. Инициализируем веса нулями
        self.weights = np.zeros(n_features)
        self.bias = 0
        # 2. Градиентный спуск
        for _ in range(self.n_iters):
            # Линейная часть: z = X * w + b
            linear_model = np.dot(X, self.weights) + self.bias
            
            # Прогоняем через сигмоиду -> получаем вероятность от 0 до 1
            y_predicted = self._sigmoid(linear_model)

            # 3. Считаем градиенты (смотри-ка, формула один в один как у линейной регрессии!)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 4. Шаг против градиента
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        # Возвращает вероятность (от 0.0 до 1.0)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        # Превращаем вероятности в жесткие классы (0 или 1) по порогу 0.5
        probabilities = self.predict_proba(X)
        return np.array([1 if p > 0.5 else 0 for p in probabilities])


# --- ТЕСТИРУЕМ НА СИНТЕТИЧЕСКИХ ДАННЫХ ---
if __name__ == "__main__":
    # Генерируем точки для двух классов (0 и 1)
    np.random.seed(42)
    X = np.random.randn(100, 2)
    # Пусть класс зависит от суммы признаков с шумом
    y = np.array([1 if (x1 + x2 > 0) else 0 for x1, x2 in X])

    # Создаем и обучаем модель
    model = CustomLogisticRegression(lr=0.1, n_iters=1000)
    model.fit(X, y)

    # Делаем предсказания
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y) * 100
    
    print("=== РЕЗУЛЬТАТЫ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ ===")
    print(f"Точность модели (Accuracy): {accuracy}%")
    print(f"Найденные веса: {np.round(model.weights, 3)}, bias: {round(model.bias, 3)}")
import numpy as np

class CustomLinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr            # Скорость обучения (Learning Rate)
        self.n_iters = n_iters  # Количество эпох/шагов
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 1. Инициализируем веса нулями
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Градиентный спуск
        for _ in range(self.n_iters):
            # Предсказание: y_pred = X * w + b
            y_predicted = np.dot(X, self.weights) + self.bias

            # Считаем градиенты (производные MSE по w и b)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Обновляем параметры (шаг против градиента)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# --- БЕНЧМАРК И ПРОВЕРКА ---
if __name__ == "__main__":
    # Генерируем синтетические данные: y = 3*X_1 + 5*X_2 + 10 + шум
    np.random.seed(42)
    X = np.random.rand(100, 2)
    true_weights = np.array([3.0, 5.0])
    true_bias = 10.0
    y = np.dot(X, true_weights) + true_bias + np.random.randn(100) * 0.1

    # Обучаем нашу модель
    model = CustomLinearRegression(lr=0.1, n_iters=2000)
    model.fit(X, y)

    print("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===")
    print(f"Истинные веса:   {true_weights}, bias: {true_bias}")
    print(f"Найденные веса: {np.round(model.weights, 3)}, bias: {np.round(model.bias, 3)}")
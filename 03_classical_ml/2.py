import numpy as np
import matplotlib.pyplot as plt
from linear_regression import CustomLinearRegression

# 1. Генерируем данные
np.random.seed(42)
X = np.random.rand(100, 1)
y = 3 * X.squeeze() + 5 + np.random.randn(100) * 0.1

# 2. СОЗДАЕМ ОБЪЕКТ МОДЕЛИ (вот этого у тебя, скорее всего, не хватает перед графиком)
# (Убедись, что твой класс называется именно CustomLinearRegression, как вы писали ранее)
model = CustomLinearRegression(lr=0.1, n_iters=1000) 

# 3. ОБУЧАЕМ ЕЕ
model.fit(X, y)

# 4. А теперь уже рисуем график и достаем веса из созданной модели!
sorted_indices = np.argsort(X[:, 0])
X_sorted = X[sorted_indices]
y_pred_sorted = np.dot(X_sorted, model.weights) + model.bias  # теперь model существует!

plt.scatter(X[:, 0], y, color='blue', alpha=0.7, label='Реальные данные')
plt.plot(X_sorted[:, 0], y_pred_sorted, color='red', linewidth=2.5, label='Предсказание модели')

plt.xlabel('Признак X')
plt.ylabel('Целевая переменная y')
plt.title('График работы CustomLinearRegression')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
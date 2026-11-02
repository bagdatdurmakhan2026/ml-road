import numpy as np
from CustomLogisticRegression import CustomLogisticRegression  # или твоя лин. регрессия
import pickle

# 1. Генерируем или подготавливаем данные для обучения
X = 2 * np.random.rand(100, 1)
y = 4 * X.squeeze() + 3 + np.random.randn(100) * 0.5

# 2. СНАЧАЛА создаем экземпляр модели
model = CustomLogisticRegression() # или имя твоего класса модели

# 3. ПОТОМ обучаем модель на данных (вызываем fit)
model.fit(X, y)

# 4. И только ТЕПЕРЬ, когда модель обучена и существует, сохраняем её через pickle!
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Модель успешно обучена и сохранена как model.pkl!")
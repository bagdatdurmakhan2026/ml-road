import numpy as np
from CustomLogisticRegression import CustomLogisticRegression
from metrics import classification_metrics

# 0. Функция train_test_split (если ты её уже писал, можно импортировать или вставить сюда)
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_len = int(X.shape[0] * test_size)
    test_indices = indices[:test_len]
    train_indices = indices[test_len:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# 1. Генерируем простейшие тестовые данные (например, 200 объектов, 2 фичи)
np.random.seed(42)
X_data = np.random.randn(200, 2)
# Создаем искусственную целевую переменную (например, сумма признаков > 0)
y_data = (X_data[:, 0] + X_data[:, 1] > 0).astype(int)

# Делим на трейст и тест
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 2. Инициализируем и обучаем модель
model = CustomLogisticRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)

# 3. Делаем предсказания на тестовой выборке
predictions = model.predict(X_test)

# 4. Считаем метрики через наш новый модуль!
results = classification_metrics(y_test, predictions)

print("=== РЕЗУЛЬТАТЫ МОЕЙ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ ===")
print("Confusion Matrix:\n", results["Confusion Matrix"])
print(f"Accuracy:  {results['Accuracy']:.4f}")
print(f"Precision: {results['Precision']:.4f}")
print(f"Recall:    {results['Recall']:.4f}")
print(f"F1-Score:  {results['F1-Score']:.4f}")
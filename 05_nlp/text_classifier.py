import numpy as np

# 1. Наш мини-датасет из текстов и векторизация (Bag-of-Words)
documents = [
    "cat loves fish",
    "dog loves meat",
    "cat and dog are friends"
]

# Собираем словарь уникальных слов
words = sorted(list(set(" ".join(documents).split())))
vectors = [[doc.split().count(word) for word in words] for doc in documents]
X = np.array(vectors)  # Форма: (3, 8) — 3 документа, 8 признаков (слов)

print(f"Размерность матрицы текстов (X): {X.shape}\n")

# 2. Наша нейросеть из модуля 04, адаптированная под размер словаря
class TextNeuralNetwork:
    def __init__(self, input_dim: int):
        # input_dim теперь равен количеству слов в словаре (8)
        self.W1 = np.random.randn(input_dim, 4) * 0.01
        self.b1 = np.zeros((1, 4))
        self.W2 = np.random.randn(4, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

# 3. Инициализируем модель (передаем размерность словаря = 8)
model = TextNeuralNetwork(input_dim=len(words))

# 4. Прогоняем текстовые векторы через нейросеть!
predictions = model.forward(X)

print("Результаты сырых предсказаний нейросети для каждого текста:")
for doc, pred in zip(documents, predictions):
    print(f"  '{doc}' -> Предсказание: {pred[0]:.4f}")
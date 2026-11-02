import numpy as np

# 1. Наш мини-датасет из текстов
documents = [
    "cat loves fish",
    "dog loves meat",
    "cat and dog are friends"
]

# 2. Собираем словарь уникальных слов (Vocabulary)
words = sorted(list(set(" ".join(documents).split())))
print(f"Словарь ({len(words)} слов): {words}\n")

# 3. Превращаем каждый текст в вектор частот слов (Bag-of-Words)
vectors = []
for doc in documents:
    doc_words = doc.split()
    # Считаем, сколько раз каждое слово из словаря встречается в предложении
    vector = [doc_words.count(word) for word in words]
    vectors.append(vector)

# Переводим в привычную нам матрицу NumPy
X = np.array(vectors)

print("Матрица признаков (X):")
print(X)
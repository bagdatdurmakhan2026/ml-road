import pandas as pd
import numpy as np
print("=== БЛОК 3: Data Engineering & Pandas — Очистка и Оптимизация ===")
# 1. Создаем "грязный" датасет (эмуляция реальных данных)
np.random.seed(42) # Фиксируем генератор случайных чисел для воспроизводимости
n_rows = 1_000_000 # Количество строк
data = {
    "user_id": np.random.randint(1000, 5000, size=n_rows), # ID пользователя
    "age": np.random.choice([np.nan, 18, 25, 34, 45, 52, 60], size=n_rows), # Возраст с пропусками
    "income": np.random.choice([np.nan, 30000, 50000, 120000, 250000], size=n_rows),# Доход с пропусками
    "category": np.random.choice(["A", "B", "C", np.nan], size=n_rows)#     Категория с пропусками
}
df = pd.DataFrame(data)
# Добавим немного дубликатов для реалистичности
df = pd.concat([df, df.iloc[:10000]], ignore_index=True) # Добавляем 10k дубликатов
print(f"Исходный размер датафрейма: {df.shape}") # 
print("\nПотребление памяти до оптимизации:")
print(df.memory_usage(deep=True) / 1024 ** 2, "MB\n") # 
# 2. Очистка данных (Data Cleaning)
# Убираем дубликаты
df.drop_duplicates(inplace=True)
# Заполняем пропуски (Imputation)
# Возраст заменим медианой, доход — средним, категорию — константой 'Unknown'
df["age"] = df["age"].fillna(df["age"].median())
df["income"] = df["income"].fillna(df["income"].mean())
df["category"] = df["category"].fillna("Unknown")
# 3. Оптимизация памяти (Memory Downcasting)
# По умолчанию Pandas берет int64 и float64. Для наших данных это избыточно.
df["user_id"] = df["user_id"].astype("uint16")
df["age"] = df["age"].astype("uint8")
df["income"] = df["income"].astype("float32")
df["category"] = df["category"].astype("category") # Категориальный тип сильно жмет память
print("Потребление памяти ПОСЛЕ оптимизации:")
print(df.memory_usage(deep=True) / 1024 ** 2, "MB\n")
# 4. Продвинутая аналитика (Aggregations)
print("=== Сводная статистика по категориям ===")
summary = df.groupby("category", observed=False).agg(
    total_users=("user_id", "count"),
    avg_income=("income", "mean"),
    median_age=("age", "median")
).reset_index()
print(summary)
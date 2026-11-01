import time
import numpy as np
print("=== 1. ТЕСТ НА СКОРОСТЬ: Цикл Python vs Векторизация NumPy ===")
size = 10_000_000
py_list_1 = list(range(size))
py_list_2 = list(range(size))
np_array_1 = np.arange(size)
np_array_2 = np.arange(size)
start_time = time.time()
py_result = [a + b for a, b in zip(py_list_1, py_list_2)]
py_time = time.time() - start_time
print("время Python (цикл): {py_time:.4f} секунд")
start_time = time.time()
np_result = np_array_1 + np_array_2
np_time = time.time() - start_time
print(f"время NumPy (векторизация): {np_time:.4f} секунд")
print(f"🚀 NumPy быстрее в {py_time / np_time:.1f} раз!\n")
print("=== 2. БРОДКАСТИНГ (Broadcasting) ===")
# Представим, что у нас есть матрица признаков: 3 объекта, 3 признака
features = np.array([[1, 2, 3], 
                     [4, 5, 6], 
                     [7, 8, 9]])
# И у нас есть вектор смещения (bias) для каждого признака
bias = np.array([10, 20, 30])
# Благодаря бродкастингу NumPy автоматически «растягивает» вектор bias 
# на каждую строку матрицы и складывает их без всяких циклов!
result_broadcast = features + bias
print("Исходная матрица признаков:\n", features)
print("Вектор bias:", bias)
print("Результат сложения через Broadcasting:\n", result_broadcast)
import numpy as np
import time

# Создаем большой массив из 1 миллиона чисел
python_list = list(range(1000000))
numpy_array = np.arange(1000000)

# Способ 1: Обычный цикл Python (медленный)
start_time = time.time()
result_list = [x * 2 for x in python_list]
print(f"Python loop time: {time.time() - start_time:.5f} seconds")

# Способ 2: Векторизация в NumPy (быстрый, на уровне Си)
start_time = time.time()
result_numpy = numpy_array * 2
print(f"NumPy vectorization time: {time.time() - start_time:.5f} seconds")
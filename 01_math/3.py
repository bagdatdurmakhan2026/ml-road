import numpy as np

# Превращаем обычные списки в массивы NumPy
matrix_a = np.array([[1, 2, 3, 6], 
                     [4, 5, 6, 5]])

matrix_b = np.array([[7, 8], 
                     [9, 1], 
                     [2, 3], 
                     [1, 2]])

# Умножение матриц делается через стандартный оператор @ или функцию np.dot()
result = matrix_a @ matrix_b ## result = np.dot(matrix_a, matrix_b) # альтернативный вариант    

print(result)
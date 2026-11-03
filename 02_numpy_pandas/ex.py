import numpy as np
a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f'Форма массива (shape): {a.sum(axis=1)}')
b = np.array([[7, 8,5]])
c = a +b
print(c)
              

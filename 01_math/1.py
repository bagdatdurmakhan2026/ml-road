# Блок 2: Линейная алгебра с нуля (без использования NumPy)
# Цель: Понять механику матричного умножения изнутри

def matrix_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Умножает матрицу A размерности (M x K) на матрицу B размерности (K x N).
    Возвращает матрицу результат размерности (M x N).
    """
    rows_a = len(A)
    cols_a = len(A[0])
    rows_b = len(B)
    cols_b = len(B[0])
    
    # Проверка на совместимость размеров матриц
    if cols_a != rows_b:
        raise ValueError(f'нельзя умножить матрицы: число столбцов первой ({cols_a}) != числу строк второй ({rows_b})!')
    
    # Инициализируем результирующую матрицу нулями (размер rows_a x cols_b)
    result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    # Классический тройной цикл (сложность O(n^3))
    # i — строка матрицы A
    # j — столбец матрицы B
    # k — индекс для поэлементного перемножения и суммирования
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += A[i][k] * B[k][j]
                
    return result


if __name__ == "__main__":
    # Тестовые матрицы
    # Матрица A: 2x3
    matrix_a = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    
    # Матрица B: 3x2
    matrix_b = [
        [7, 8],
        [9, 1],
        [2, 3]
    ]
    
    print("=== Умножение матриц на чистом Python ===")
    print("Матрица A:")
    for row in matrix_a:
        print(row)
        
    print("\nМатрица B:")
    for row in matrix_b:
        print(row)
        
    res = matrix_multiply(matrix_a, matrix_b)
    
    print("\nРезультат (A x B):")
    for row in res:
        print(row)
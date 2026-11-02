def ans(A, B):
    a = len(A)
    b = len(A[0])
    c = len(B)
    d = len(B[0])
    if b != c:
        raise ValueError(f'нельзя умножить матрицы: число столбцов первой ({b}) != числу строк второй ({c})!')
    result = [[0.0 for _ in range(d)] for _ in range(a)]
    for i in range(a):
        for j in range(d):
            for k in range(b):
                result[i][j] += A[i][k] * B[k][j]
    return result
if __name__ == "__main__":
    matrix_a = [
        [1, 2, 3,6 ],
        [4, 5, 6, 5]
    ]
    matrix_b = [
        [7, 8],
        [9, 1],
        [2, 3],
        [1, 2]
    ]
    for row in ans(matrix_a, matrix_b):
        print(row)
import numpy as np
class LinearRegression:
    def __init__(self, lr= 0.01, n_iters= 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
    def ans(self, X,y):
        self.X = X
        ns, nf = X.shape
        self.w = np.zeros(nf)
        self.b = 0
        for _ in range(self.n_iters):
            yp = X @ self.w + self.b
            dw = (1/ns) * X.T @ (yp - y)
            db = (1/ns) * np.sum(yp - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
    def predict(self, X):
        return X @ self.w + self.b
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(300, 3)
    tw = np.array([2.0, -1.0, 3.0])
    tb = 5.0
    y = X @ tw + tb + np.random.randn(300) * 0.1
    model = LinearRegression(lr=0.01, n_iters=1000)
    model.ans(X, y)
    predictions = model.predict(X)
    print("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===")
    print(f"Истинные веса:   {tw}, bias: {tb}")
    print(f"Найденные веса: {np.round(model.w, 3)}, bias: {np.round(model.b, 3)}")
    
import numpy as np

class TrainingNeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(3, 4) * 0.01
        self.b1 = np.zeros((1, 4))
        self.W2 = np.random.randn(4, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, lr: float = 0.01):
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = np.mean((predictions - y) ** 2)
            
            dZ2 = 2 * (predictions - y) / X.shape[0]
            dW2 = np.dot(self.A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * (self.Z1 > 0)  # по производной ReLU
            
            dW1 = np.dot(X.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            
            if (epoch + 1) % 200 == 0:
                print(f"Эпоха {epoch + 1}/{epochs}, Loss (Ошибка): {loss:.6f}")

if __name__ == "__main__":
    X_train = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 0.0],
        [1.5, 1.5, 1.5],
        [0.5, 0.5, 0.5]
    ])
    y_train = np.array([[12.0], [3.0], [9.0], [3.0]])
    
    net = TrainingNeuralNetwork()
    print("Запуск обучения нейросети...")
    net.train(X_train, y_train, epochs=1000, lr=0.01)
    
    test_sample = np.array([[1.0, 2.0, 3.0]])
    print("Прогноз для [1.0, 2.0, 3.0] после обучения:", net.forward(test_sample))
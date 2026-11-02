import numpy as np

class SimpleNeuralNetwork:
    def __init__(self):
        # Первый скрытый слой: 3 входа -> 4 нейрона
        self.layer1_weights = np.random.randn(3, 4) * 0.01
        self.layer1_biases = np.zeros((1, 4))
        
        # Второй выходной слой: 4 входа -> 1 нейрон
        self.layer2_weights = np.random.randn(4, 1) * 0.01
        self.layer2_biases = np.zeros((1, 1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Прямой проход через первый слой + ReLU
        self.z1 = np.dot(X, self.layer1_weights) + self.layer1_biases
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        # Прямой проход через второй (выходной) слой
        self.z2 = np.dot(self.a1, self.layer2_weights) + self.layer2_biases
        return self.z2

# Тестируем многослойную сеть
if __name__ == "__main__":
    X = np.array([[1.0, 2.0, 3.0]])  # Входные данные (1 образец, 3 признака)
    model = SimpleNeuralNetwork()
    
    prediction = model.forward(X)
    print("Финальный прогноз нейросети:", prediction)
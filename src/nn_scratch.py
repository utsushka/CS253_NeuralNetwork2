import numpy as np
import pickle
import os


class ScratchNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Инициализация весов (He initialization для ReLU)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    # === ИЗМЕНЕНИЕ 1: Замена сигмоиды на ReLU ===
    def relu(self, x):
        """Функция активации ReLU."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Производная ReLU (0 для x < 0, 1 для x > 0)."""
        return (x > 0).astype(float)

    # === Конец изменений ===

    def softmax(self, x):
        """Функция активации Softmax для выходного слоя."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """Прямой проход."""
        # Flatten input если нужно
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        self.z1 = np.dot(X, self.W1) + self.b1

        # === ИЗМЕНЕНИЕ 2: Используем ReLU вместо sigmoid ===
        self.a1 = self.relu(self.z1)  # Было: self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_one_hot, output):
        """Обратный проход."""
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        m = X.shape[0]

        # Output layer error (cross-entropy + softmax derivative)
        dz2 = output - y_one_hot
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer error
        # === ИЗМЕНЕНИЕ 3: Используем производную ReLU ===
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.a1)
        # Было: * self.sigmoid_derivative(self.a1)

        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=10):
        """Обучение сети."""
        # One-hot encoding
        y_one_hot = np.zeros((y.size, self.output_size))
        y_one_hot[np.arange(y.size), y] = 1

        print("Начало обучения (Scratch NN) с ReLU...")
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y_one_hot, output)

            if i % 10 == 0:
                loss = -np.mean(y_one_hot * np.log(output + 1e-9))
                acc = np.mean(np.argmax(output, axis=1) == y)
                print(f"Epoch {i}, Loss: {loss:.4f}, Acc: {acc:.4f}")

    def predict(self, X):
        """Предсказание."""
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def save(self, path):
        """Сохранение модели."""
        with open(path, 'wb') as f:
            pickle.dump({'W1': self.W1, 'b1': self.b1,
                         'W2': self.W2, 'b2': self.b2}, f)

    def load(self, path):
        """Загрузка модели."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.W1 = data['W1']
                self.b1 = data['b1']
                self.W2 = data['W2']
                self.b2 = data['b2']
            return True
        return False
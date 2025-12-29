import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

class LibraryNN:
    def __init__(self, input_shape, num_classes):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Reshape(input_shape + (1,)),  # Добавляем канал цвета
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, X, y, epochs=10, validation_data=None):
        print("Начало обучения (Keras NN)...")
        # Аугментация данных "на лету"
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        # Reshape for CNN: (N, 32, 32, 1)
        X_reshaped = X.reshape(-1, 32, 32, 1)

        if validation_data:
            X_val, y_val = validation_data
            X_val = X_val.reshape(-1, 32, 32, 1)
            val_data = (X_val, y_val)
        else:
            val_data = None

        self.model.fit(datagen.flow(X_reshaped, y, batch_size=32),
                       epochs=epochs,
                       validation_data=val_data)

    def predict(self, X):
        X_reshaped = X.reshape(-1, 32, 32, 1)
        probs = self.model.predict(X_reshaped, verbose=0)
        return np.argmax(probs, axis=1)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        if os.path.exists(path):
            self.model = models.load_model(path)
            return True
        return False
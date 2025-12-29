import os
import tkinter as tk
from src.data_loader import load_data
from src.nn_scratch import ScratchNN
from src.nn_library import LibraryNN
from src.gui import App

DATA_PATH = os.path.join('data', 'Working Dataset_Split (70-15-15)')
MODEL_DIR = 'models'
SCRATCH_MODEL_PATH = os.path.join(MODEL_DIR, 'scratch_model.pkl')
LIB_MODEL_PATH = os.path.join(MODEL_DIR, 'lib_model.keras')

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    (X_train, y_train), (X_test, y_test), classes = load_data(DATA_PATH)

    input_size = 32 * 32
    num_classes = len(classes)

    scratch_net = ScratchNN(input_size, 128, num_classes, learning_rate=0.05)
    lib_net = LibraryNN((32, 32), num_classes)

    print("--- Проверка моделей ---")

    # --- Самописная сеть ---
    if scratch_net.load(SCRATCH_MODEL_PATH):
        print("Загружена самописная модель.")
    else:
        print("Обучение самописной модели...")
        scratch_net.train(X_train, y_train, epochs=200)
        scratch_net.save(SCRATCH_MODEL_PATH)

    # Тест самописной
    preds = scratch_net.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"Точность самописной сети на тесте: {acc:.2%}")

    # --- Библиотечная сеть ---
    if lib_net.load(LIB_MODEL_PATH):
        print("Загружена библиотечная модель.")
    else:
        print("Обучение библиотечной модели...")
        lib_net.train(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
        lib_net.save(LIB_MODEL_PATH)

    # Тест библиотечной
    preds_lib = lib_net.predict(X_test)
    acc_lib = (preds_lib == y_test).mean()
    print(f"Точность библиотечной сети на тесте: {acc_lib:.2%}")

    # Запуск GUI
    print("Запуск графического интерфейса...")
    root = tk.Tk()
    app = App(root, scratch_net, lib_net, classes)
    root.mainloop()

if __name__ == "__main__":
    main()
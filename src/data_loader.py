import os
import cv2
import numpy as np
from .preprocessor import process_image

# Выберем 10 классов греческих букв
CLASSES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
           'zeta', 'eta', 'theta', 'iota', 'kappa']

def load_data(base_path):
    """
    Загружает Train и Test выборки.
    Структура base_path должна содержать папки Train, Test (и Val).
    """
    print(f"Загрузка данных из {base_path}...")

    X_train, y_train = _load_folder(os.path.join(base_path, 'Train'))
    X_test, y_test = _load_folder(os.path.join(base_path, 'Test'))

    print(f"Загружено: Train {len(X_train)}, Test {len(X_test)}")
    return (X_train, y_train), (X_test, y_test), CLASSES

def _load_folder(folder_path):
    X = []
    y = []

    for idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(folder_path, class_name)
        # Попробуем найти папку нечувствительно к регистру (alpha, Alpha, ALPHA)
        actual_folders = os.listdir(folder_path)
        found_folder = None
        for f in actual_folders:
            if f.lower() == class_name.lower():
                found_folder = f
                break

        if not found_folder:
            print(f"Предупреждение: папка {class_name} не найдена.")
            continue

        full_path = os.path.join(folder_path, found_folder)
        files = os.listdir(full_path)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(full_path, file)
                img = cv2.imread(img_path)
                if img is None: continue

                # Предобработка

                # Инвертируем цвет, превращаем черную букву на белом фоне
                # в белую букву на черном фоне
                processed = process_image(img, invert=True)
                X.append(processed)
                y.append(idx)  # Метка класса - число

    return np.array(X), np.array(y)
import cv2
import numpy as np

IMG_SIZE = 32  # Размер входного изображения для сети

def process_image(img_array, invert=False):
    """
    Принимает изображение (numpy array), превращает в ч/б,
    находит границы символа, обрезает и масштабирует к IMG_SIZE x IMG_SIZE.
    """
    # 1. Перевод в оттенки серого
    if len(img_array.shape) > 2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array

    # 2. Бинаризация и инверсия
    if invert:
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    else:
        # Если датасет уже белый на черном, или наоборот, подстраиваемся.
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 3. Поиск контуров для обрезки лишних полей
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros((IMG_SIZE, IMG_SIZE))  # Пустой квадрат если ничего не нашли

    # Находим самый большой контур (предполагаем, что это буква)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Обрезаем (ROI - Region of Interest)
    roi = thresh[y:y + h, x:x + w]

    # 4. Масштабирование с сохранением пропорций
    h_roi, w_roi = roi.shape
    scale = (IMG_SIZE - 4) / max(h_roi, w_roi)  # -4 для отступа
    new_w = int(w_roi * scale)
    new_h = int(h_roi * scale)

    resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5. Размещение по центру квадрата IMG_SIZE x IMG_SIZE
    final_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    start_x = (IMG_SIZE - new_w) // 2
    start_y = (IMG_SIZE - new_h) // 2
    final_img[start_y:start_y + new_h, start_x:start_x + new_w] = resized_roi

    # Нормализация 0..1
    return final_img / 255.0
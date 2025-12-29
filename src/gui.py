import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
from .preprocessor import process_image

class App:
    def __init__(self, root, scratch_model, lib_model, classes):
        self.root = root
        self.root.title("Распознавание греческих букв")

        self.scratch_model = scratch_model
        self.lib_model = lib_model
        self.classes = classes

        # Веб-камера
        self.cap = cv2.VideoCapture(0)

        # UI Элементы
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=10, pady=10)

        # Видео
        self.lmain = tk.Label(self.main_frame)
        self.lmain.grid(row=0, column=0, columnspan=2)

        # Кнопка захвата
        self.btn_capture = tk.Button(self.main_frame, text="Распознать", command=self.capture_and_predict,
                                     font=("Arial", 14))
        self.btn_capture.grid(row=1, column=0, columnspan=2, pady=10)

        # Результаты
        self.res_frame = tk.LabelFrame(self.main_frame, text="Результаты", font=("Arial", 12))
        self.res_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        self.lbl_scratch = tk.Label(self.res_frame, text="Самописная сеть: ...", font=("Arial", 12), fg="blue")
        self.lbl_scratch.pack(anchor="w", padx=5)

        self.lbl_lib = tk.Label(self.res_frame, text="Библиотечная сеть: ...", font=("Arial", 12), fg="green")
        self.lbl_lib.pack(anchor="w", padx=5)

        # Предпросмотр того, что видит сеть
        self.lbl_debug = tk.Label(self.main_frame, text="Вход сети:")
        self.lbl_debug.grid(row=3, column=0)
        self.preview_label = tk.Label(self.main_frame, bg="black")
        self.preview_label.grid(row=3, column=1)

        self.video_loop()

    def video_loop(self):
        ret, frame = self.cap.read()
        if ret:
            # Рисуем прямоугольник, куда пользователю писать букву
            h, w, _ = frame.shape
            cv2.rectangle(frame, (w // 2 - 100, h // 2 - 100), (w // 2 + 100, h // 2 + 100), (0, 255, 0), 2)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)
        self.root.after(10, self.video_loop)

    def capture_and_predict(self):
        ret, frame = self.cap.read()
        if ret:
            h, w, _ = frame.shape
            # Вырезаем центральную область
            crop = frame[h // 2 - 100:h // 2 + 100, w // 2 - 100:w // 2 + 100]

            # Обработка (инвертируем цвет)
            processed = process_image(crop, invert=True)

            # Показать, что видит сеть
            debug_img = Image.fromarray((processed * 255).astype(np.uint8))
            debug_img = debug_img.resize((100, 100), Image.NEAREST)
            debug_tk = ImageTk.PhotoImage(image=debug_img)
            self.preview_label.imgtk = debug_tk
            self.preview_label.configure(image=debug_tk)

            # Подготовка для сети (добавляем batch dimension)
            input_data = np.expand_dims(processed, axis=0)

            # Предсказание Scratch
            pred_s = self.scratch_model.predict(input_data)[0]
            self.lbl_scratch.config(text=f"Самописная сеть: {self.classes[pred_s]}")

            # Предсказание Library
            pred_l = self.lib_model.predict(input_data)[0]
            self.lbl_lib.config(text=f"Библиотечная сеть: {self.classes[pred_l]}")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
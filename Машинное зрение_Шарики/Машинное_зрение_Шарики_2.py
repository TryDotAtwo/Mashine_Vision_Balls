import os
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"
os.environ["QT_PLUGIN_PATH"] = r"C:\Users\Иван Литвак\AppData\Local\Programs\Python\Python311\Lib\site-packages\PyQt5\Qt5\plugins"

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

def process_video_fast(video_path):
    """
    Быстрая обработка видео с отображением этапов через PyQtGraph с оптимизацией для уменьшения лагов.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Ошибка при открытии видеофайла")

    # Инициализация PyQtGraph
    app = pg.mkQApp()
    win = pg.GraphicsLayoutWidget(title="Обработка видео")
    win.resize(1280, 720)
    win.show()

    # Определение расположения этапов
    window_layout = {
        'Оригинал': (0, 0),
        'Грэйскейл': (1, 0),
        'После фильтрации': (2, 0),
        'Вычитание фона': (3, 0),
        'Размытие освещения': (0, 1),
        'Высокие частоты': (1, 1),
        'Бинаризация': (2, 1),
        'Морфология': (3, 1),
        'Наложение': (0, 2)
    }

    # Создание ImageItem для каждого этапа
    images = {}
    for name, (col, row) in window_layout.items():
        view = win.addViewBox(col=col, row=row)
        img_item = pg.ImageItem()
        view.addItem(img_item)
        view.setAspectLocked(True)
        view.setRange(xRange=(0, 320), yRange=(0, 240), padding=0)  # Ограничение области отображения
        images[name] = img_item
        win.nextRow() if col == 3 and row < 2 else None

    # Инициализация фона
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Не удалось прочитать первый кадр")
    cropped = frame[:, :1100]
    cropped = cv2.resize(cropped, (320, 240))  # Уменьшение разрешения
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 100, 100)
    background = np.median(filtered).astype(np.float32)
    alpha = 0.1

    def update_frame():
        nonlocal background
        ret, frame = cap.read()
        if not ret:
            app.quit()
            return

        # Шаги обработки
        cropped = frame[:, :1100]
        cropped = cv2.resize(cropped, (320, 240))  # Уменьшение разрешения
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 50, 50)
        background = background * (1 - alpha) + np.median(filtered) * alpha
        fg = np.clip(filtered - background, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(fg, (21, 21), 0)
        high_pass = cv2.addWeighted(fg, 1.5, blurred, -0.5, 0)
        binary = cv2.adaptiveThreshold(
            high_pass, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        color = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        mask_color = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(color, 0.7, mask_color, 0.3, 0)

        # Отображение через PyQtGraph
        images['Оригинал'].setImage(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), levels=(0, 255))
        images['Грэйскейл'].setImage(gray, levels=(0, 255))
        images['После фильтрации'].setImage(filtered, levels=(0, 255))
        images['Вычитание фона'].setImage(fg, levels=(0, 255))
        images['Размытие освещения'].setImage(blurred, levels=(0, 255))
        images['Высокие частоты'].setImage(high_pass, levels=(0, 255))
        images['Бинаризация'].setImage(binary, levels=(0, 255))
        images['Морфология'].setImage(morph, levels=(0, 255))
        images['Наложение'].setImage(overlay, levels=(0, 255))

        app.processEvents()

    # Таймер для обновления кадров
    timer = QtCore.QTimer()
    timer.timeout.connect(update_frame)
    timer.start(int(1000/60))  # ~30 FPS (1000 / 30)

    # Запуск приложения
    if __name__ == "__main__":
        pg.exec()

    # Освобождение ресурсов
    cap.release()

if __name__ == "__main__":
    process_video_fast(r'D:\Шарыкы\Для машинного зрения\20250506_150457.mp4')
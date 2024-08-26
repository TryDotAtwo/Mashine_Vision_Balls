import cv2
import numpy as np

def count_spherical_objects(video_path):
    # Загрузка видео
    cap = cv2.VideoCapture(video_path)

    while True:
        # Чтение текущего кадра
        ret, frame = cap.read()

        if not ret:
            break

        # Преобразование кадра в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Увеличение яркости кадра
        brightened = cv2.addWeighted(gray, 1.2, np.zeros_like(gray), 0, 0)

        # Применение гауссовского размытия для сглаживания кадра
        blurred = cv2.GaussianBlur(brightened, (5, 5), 0)

        # Обнаружение кругов на кадре
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=10, maxRadius=50)

        # Подсчет и отображение сферических объектов на кадре
        if circles is not None:
            total_objects = len(circles[0])
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 1)

            cv2.putText(frame, f"Total Objects: {total_objects}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Отображение текущего кадра с выделенными контурами и информацией о числе обнаруженных объектов
        cv2.imshow("Spherical Objects", frame)

        # Выход из цикла при нажатии клавиши "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

# Путь к видео файлу
video_path = 'D:\Ball3.mp4'

# Подсчет и отображение сферических объектов на видео
count_spherical_objects('D:\Ball.mp4')
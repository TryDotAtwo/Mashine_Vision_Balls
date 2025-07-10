import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch.cuda as cuda
import cv2
from screeninfo import get_monitors
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def execute_sharik_v_unitaze():
    # print(torch.cuda.is_available())

    # Получаем разрешение экрана
    monitor = get_monitors()[0]  # Получаем информацию о первом мониторе
    screen_width = monitor.width
    screen_height = monitor.height
    x, y, w, h = 700, 1500, 700, 700  # Координаты обрезки видео

    # Путь к видеофайлу
    video_path = r'D:/Ball12.mp4'

    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка при открытии видеофайла")
        exit()
    # Будем выводить на график колебания радиуса на каждом кадре
    frame_numbers = []
    radius_in_pixels = []
    frame_n = 0

    while True:

        # Чтение кадра из видео
        ret, frame = cap.read()

        if not ret:
            print("Видео завершено")
            break

        # Вырезаем квадратик
        square = frame[y:y + h, x:x + w]

        gray_frame = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 2)

        # # Распознаем круги на монохромном кадре
        circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                                   param1=100, param2=50, minRadius=50, maxRadius=100)

        # Если были найдены круги, рисуем их на оригинальном кадре, и подбираем их точный радиус
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Создаем черный маску того же размера, что и изображение
                height, width, _ = square.shape
                ball_x, ball_y, ball_r = i[0], i[1], i[2]
                cut_mask = np.zeros((height, width), dtype=np.uint8)

                # Рисуем круг на маске по которому вырежем изображение шарика
                cut_circle_thickness = 24
                cv2.circle(cut_mask, (ball_x, ball_y), ball_r, (255, 255, 255), cut_circle_thickness)

                # Вырезаем круг из изображения
                cut_out_circle = cv2.bitwise_and(square, square, mask=cut_mask)
                # cv2.imshow('Cut', cut_out_circle)

                white_circle_thickness = cut_circle_thickness // 2

                white_circle_radius = ball_r - white_circle_thickness // 2
                # cv2.imshow('White', white_circle)

                # Вырезаем область за пределами круга
                # inverse_mask = cv2.bitwise_not(cut_mask)
                # background = cv2.bitwise_and(square, square, mask=inverse_mask)

                cut_out_gray = cv2.cvtColor(cut_out_circle, cv2.COLOR_BGR2GRAY)
                cv2.imshow('test', cut_out_gray)

                # Провариируем радиус белого кружка
                variate_steps = 5
                sim_index_and_radius = {}
                for i in range(variate_steps):
                    # Рисуем белый круг на маске с которым будем сравнивать вырезанный круг, он меньше вырезанного, так шоб их внутренние
                    # диаметры были одинаковы, а наружный диаметр будем вариировать

                    white_circle = np.zeros((height, width), dtype=np.uint8)
                    cv2.circle(white_circle, (ball_x, ball_y), white_circle_radius, (255, 255, 255),
                               white_circle_thickness)
                    white_circle = cv2.GaussianBlur(white_circle, (3, 3), 2)

                    # Вычисляем SSIM
                    similarity_index, _ = ssim(cut_out_gray, white_circle, full=True)
                    # print(f"Индекс подобия (SSIM): {similarity_index:.4f}")
                    sim_index_and_radius[i] = {similarity_index, white_circle_radius}
                    white_circle_radius += 1
                    max_similarity = 0
                    radius = 0
                    for ind, pairs in sim_index_and_radius.items():
                        # print(pairs)
                        iterator = iter(pairs)
                        ms = next(iterator)
                        rad = next(iterator)
                        if ms > max_similarity:
                            max_similarity = ms
                            radius = rad

                    frame_numbers.append(frame_n)
                    frame_n += 1
                    radius_in_pixels.append(radius)
                    cv2.circle(square, (ball_x, ball_y), radius, (0, 255, 0), 3)
                    cv2.imshow('test', square)  

            # print(max_similarity)
            # print(radius)
            # print('-----')

            # Получаем размеры кадра
            frame_height, frame_width = gray_frame.shape

            # Вычисляем коэффициент масштабирования по ширине и высоте
            scale_width = screen_width / frame_width
            scale_height = screen_height / frame_height
            scale = min(scale_width, scale_height)

            # Масштабируем кадр под экран
            new_width = int(frame_width * scale - 150)
            new_height = int(frame_height * scale - 150)
            resized_frame = cv2.resize(gray_frame, (new_width, new_height))

            # Отображаем кадр
            # cv2.imshow('Sharik v unitaze', resized_frame)

            # Ожидаем 25 мс, чтобы показать следующий кадр
            # Нажмите 'q', чтобы выйти
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    plt.plot(frame_numbers, radius_in_pixels, marker='o')
    plt.title('График радиус от номера кадра')
    plt.xlabel('Номер кадра')
    plt.ylabel('Радиус')
    plt.grid()

    # Показываем график
    plt.show()

    # Освобождаем захват видео и закрываем окна
    cap.release()
cv2.destroyAllWindows()


execute_sharik_v_unitaze()
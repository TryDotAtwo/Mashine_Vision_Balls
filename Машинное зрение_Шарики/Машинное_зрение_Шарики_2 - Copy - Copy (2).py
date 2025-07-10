import numpy as np
import cupy as cp
import cv2
from screeninfo import get_monitors
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def ssim_gpu(img1, img2):
    """
    Вычисление SSIM на GPU с использованием PyTorch.
    """
    img1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0
    img2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float().cuda() / 255.0

    mu1 = F.conv2d(img1, torch.ones(1, 1, 3, 3).cuda() / 9, padding=1)
    mu2 = F.conv2d(img2, torch.ones(1, 1, 3, 3).cuda() / 9, padding=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, torch.ones(1, 1, 3, 3).cuda() / 9, padding=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, torch.ones(1, 1, 3, 3).cuda() / 9, padding=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, torch.ones(1, 1, 3, 3).cuda() / 9, padding=1) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def execute_sharik_v_unitaze():
    # Получаем разрешение экрана
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    x, y, w, h = 700, 1500, 700, 700  # Координаты обрезки видео

    # Путь к видеофайлу
    video_path = r'D:\Ball12.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка при открытии видеофайла")
        exit()

    frame_numbers = []
    radius_in_pixels = []
    optimal_centers = []
    frame_n = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Видео завершено")
            break

        square = frame[y:y + h, x:x + w]

        gray_frame = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 2)

        # Распознаем круги с помощью Hough Circles
        circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                                   param1=100, param2=50, minRadius=50, maxRadius=100)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                height, width, _ = square.shape
                ball_x, ball_y, ball_r = i[0], i[1], i[2]

                cut_mask = np.zeros((height, width), dtype=np.uint8)
                cut_circle_thickness = 24
                cv2.circle(cut_mask, (ball_x, ball_y), ball_r, (255, 255, 255), cut_circle_thickness)

                cut_out_circle = cv2.bitwise_and(square, square, mask=cut_mask)
                cut_out_gray = cv2.cvtColor(cut_out_circle, cv2.COLOR_BGR2GRAY)

                # Конвертируем данные в cupy для работы на GPU
                cut_out_gray_gpu = cp.array(cut_out_gray)

                # cv2.imshow('test', cut_out_gray)
                variate_steps = 5
                center_variate_steps = 50
                max_similarity = 0
                optimal_radius = 0
                optimal_center = (ball_x, ball_y)

                for dx in range(-center_variate_steps, center_variate_steps + 1):
                    for dy in range(-center_variate_steps, center_variate_steps + 1):
                        for i in range(variate_steps):
                            white_circle = cp.zeros((height, width), dtype=cp.uint8)

                            # Преобразование white_circle из cuPy в NumPy перед использованием в OpenCV
                            white_circle_numpy = cp.asnumpy(white_circle)       

                            current_center_x = ball_x + dx
                            current_center_y = ball_y + dy

                            white_circle_radius = ball_r - cut_circle_thickness // 2
                            cv2.circle(white_circle_numpy, (current_center_x, current_center_y),
                                       white_circle_radius, (255, 255, 255), cut_circle_thickness // 2)

                            # Преобразование обратно в cuPy, если необходимо продолжить дальнейшие вычисления на GPU
                            white_circle = cp.asarray(white_circle_numpy)

                            white_circle = cp.asarray(cv2.GaussianBlur(cp.asnumpy(white_circle), (3, 3), 2))

                            # Вычисляем SSIM на GPU
                            similarity_index = ssim_gpu(cp.asnumpy(cut_out_gray_gpu), cp.asnumpy(white_circle))

                            if similarity_index > max_similarity:
                                max_similarity = similarity_index
                                optimal_radius = white_circle_radius
                                optimal_center = (current_center_x, current_center_y)
                              

                frame_numbers.append(frame_n)
                radius_in_pixels.append(optimal_radius)
                optimal_centers.append(optimal_center)
                frame_n += 1
                cv2.circle(cut_out_gray, (optimal_center[0], optimal_center[1]), optimal_radius, (0, 255, 0), 3)
                cv2.imshow('test', cut_out_gray)  
                

        # Масштабируем кадр под экран
        frame_height, frame_width = gray_frame.shape
        scale_width = screen_width / frame_width
        scale_height = screen_height / frame_height
        scale = min(scale_width, scale_height)

        new_width = int(frame_width * scale - 150)
        new_height = int(frame_height * scale - 150)
        resized_frame = cv2.resize(gray_frame, (new_width, new_height))

        # Отображаем кадр
        # cv2.imshow('Sharik v unitaze', resized_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    plt.plot(frame_numbers, radius_in_pixels, marker='o')
    plt.title('График радиус от номера кадра')
    plt.xlabel('Номер кадра')
    plt.ylabel('Радиус')
    plt.grid()
    plt.show()

    cap.release()
    cv2.destroyAllWindows()

execute_sharik_v_unitaze()

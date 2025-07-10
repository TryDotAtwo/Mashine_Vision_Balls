import os
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"
os.environ["QT_PLUGIN_PATH"] = r"C:\Users\Иван Литвак\AppData\Local\Programs\Python\Python311\Lib\site-packages\PyQt5\Qt5\plugins"

import numpy as np
import cv2
from screeninfo import get_monitors
import torch
import torch.nn.functional as F
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

# Dictionary for plot labels in Russian and English
PLOT_LABELS = {
    'ru': {
        'dist_title': 'Распределение радиусов шариков',
        'dist_x': 'Радиус (пиксели)',
        'dist_y': 'Частота',
        'curr_pos_title': 'Положение текущего шарика',
        'curr_pos_x': 'X (пиксели)',
        'curr_pos_y': 'Y (пиксели)',
        'curr_pos_traj': 'Траектория',
        'prev_pos_title': 'Положение предыдущего шарика',
        'prev_pos_x': 'X (пиксели)',
        'prev_pos_y': 'Y (пиксели)',
        'prev_pos_traj': 'Траектория',
        'speed_title': 'Скорость шариков',
        'speed_x': 'Время (с)',
        'speed_y': 'Скорость (мм/с)',
        'speed_curr': 'Текущий',
        'speed_prev': 'Предыдущий',
        'radius_title': 'Размер шарика',
        'radius_x': 'Время (с)',
        'radius_y': 'Радиус (пиксели)',
        'radius_curr': 'Текущий',
        'radius_mean': 'Скользящее среднее',
        'radius_prev': 'Предыдущий',
        'window_title': 'Графики'
    },
    'en': {
        'dist_title': 'Distribution of Ball Radii',
        'dist_x': 'Radius (pixels)',
        'dist_y': 'Frequency',
        'curr_pos_title': 'Current Ball Position',
        'curr_pos_x': 'X (pixels)',
        'curr_pos_y': 'Y (pixels)',
        'curr_pos_traj': 'Trajectory',
        'prev_pos_title': 'Previous Ball Position',
        'prev_pos_x': 'X (pixels)',
        'prev_pos_y': 'Y (pixels)',
        'prev_pos_traj': 'Trajectory',
        'speed_title': 'Ball Speed',
        'speed_x': 'Time (s)',
        'speed_y': 'Speed (mm/s)',
        'speed_curr': 'Current',
        'speed_prev': 'Previous',
        'radius_title': 'Ball Size',
        'radius_x': 'Time (s)',
        'radius_y': 'Radius (pixels)',
        'radius_curr': 'Current',
        'radius_mean': 'Moving Average',
        'radius_prev': 'Previous',
        'window_title': 'Plots'
    }
}

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

def find_ball(gray_frame, current_ball, current_time, last_avg_speed, config):
    """
    Поиск шарика с помощью HoughCircles и проверка скорости.
    Возвращает: (ball_x, ball_y, ball_r) или None, если шарик не найден.
    """
    circles = None
    retry_count = 0
    while retry_count < config['MAX_RETRIES']:
        circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, dp=config['HOUGH_DP'], 
                                   minDist=config['HOUGH_MIN_DIST'], param1=config['HOUGH_PARAM1'], 
                                   param2=config['HOUGH_PARAM2'], minRadius=config['HOUGH_MIN_RADIUS'], 
                                   maxRadius=config['HOUGH_MAX_RADIUS'])
        
        if circles is None:
            retry_count += 1
            continue

        circles = np.uint16(np.around(circles))
        i = circles[0, 0]
        ball_x, ball_y, ball_r = int(i[0]), int(i[1]), int(i[2])

        if len(current_ball['centers']) < 5:
            return ball_x, ball_y, ball_r

        if len(current_ball['centers']) >= 5:
            prev_center = np.array(current_ball['centers'][-1])
            curr_center = np.array((ball_x, ball_y))
            time_diff = current_time - current_ball['times'][-1]
            if time_diff == 0:
                time_diff = 1e-6
            speed = np.linalg.norm(curr_center - prev_center) / time_diff

            avg_speed = np.mean(current_ball['speeds'][-5:]) if len(current_ball['speeds']) >= 5 else last_avg_speed
            if avg_speed == 0:
                avg_speed = 1.0

            if speed > config['SPEED_THRESHOLD_MULTIPLIER'] * avg_speed:
                retry_count += 1
                circles = None
                print(f"Повторная попытка из-за высокой скорости: {speed:.2f}")
                continue

        return ball_x, ball_y, ball_r

    print("Все попытки исчерпаны, шарик не найден")
    return None

def adjust_ball(square, gray_frame, ball_x, ball_y, ball_r, frame_height, frame_width, config):
    """
    Корректировка радиуса и положения шарика через SSIM.
    Возвращает: optimal_center, optimal_radius.
    """
    cut_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.circle(cut_mask, (ball_x, ball_y), ball_r + config['CUT_MASK_OFFSET'], (255, 255, 255), -1)

    cut_out_circle = cv2.bitwise_and(square, square, mask=cut_mask)
    cut_out_gray = cv2.cvtColor(cut_out_circle, cv2.COLOR_BGR2GRAY)
    cv2.circle(cut_out_gray, (ball_x, ball_y), ball_r - config['CUT_MASK_OFFSET'], (255, 255, 255), -1)

    max_similarity = 0
    optimal_radius = ball_r
    optimal_center = (ball_x, ball_y)

    for dx in range(-config['CENTER_VARIATE_STEPS'], config['CENTER_VARIATE_STEPS'] + 1):
        for dy in range(-config['CENTER_VARIATE_STEPS'], config['CENTER_VARIATE_STEPS'] + 1):
            for i in range(config['VARIATE_STEPS']):
                white_circle = np.zeros((frame_height, frame_width), dtype=np.uint8)
                leght = -1

                current_center_x = ball_x + dx
                current_center_y = ball_y + dy

                white_circle_radius = int(ball_r + i)
                cv2.circle(white_circle, (current_center_x, current_center_y), white_circle_radius, (255, 255, 255), leght)
                cv2.GaussianBlur(white_circle, (config['GAUSSIAN_KERNEL_SIZE'], config['GAUSSIAN_KERNEL_SIZE']), config['GAUSSIAN_SIGMA'])

                similarity_index = ssim_gpu(cut_out_gray, white_circle)
                if similarity_index > max_similarity:
                    max_similarity = similarity_index
                    optimal_radius = white_circle_radius
                    optimal_center = (current_center_x, current_center_y)

                white_circle = np.zeros((frame_height, frame_width), dtype=np.uint8)
                white_circle_radius = int(ball_r - i)
                cv2.circle(white_circle, (current_center_x, current_center_y), white_circle_radius, (255, 255, 255), leght)
                cv2.GaussianBlur(white_circle, (config['GAUSSIAN_KERNEL_SIZE'], config['GAUSSIAN_KERNEL_SIZE']), config['GAUSSIAN_SIGMA'])

                similarity_index = ssim_gpu(cut_out_gray, white_circle)
                if similarity_index > max_similarity:
                    max_similarity = similarity_index
                    optimal_radius = white_circle_radius
                    optimal_center = (current_center_x, current_center_y)

    return optimal_center, optimal_radius

def check_ball_status(current_ball, no_ball_frames, config):
    """
    Проверяет, утонул ли шарик или алгоритм ошибся.
    Возвращает: True, если шарик утонул, иначе False.
    """
    if no_ball_frames >= config['MAX_NO_BALL_FRAMES'] and current_ball['times'] and \
       (current_ball['times'][-1] - current_ball['times'][0]) >= config['MIN_BALL_DURATION']:
        return True
    return False

def update_plots(current_ball, previous_ball, ball_avg_radii, video_img, cut_out_gray, 
                 curve_curr_pos, curve_prev_pos, curve_curr_speed, curve_prev_speed, 
                 curve_curr_radius, curve_curr_mean_radius, curve_prev_radius, 
                 plot_dist, plot_curr_pos, plot_speed, plot_prev_pos, plot_radius, config, lang):
    """
    Обновляет все графики и видео с использованием PyQtGraph.
    """
    labels = PLOT_LABELS[lang]
    
    # График текущей траектории с градиентом цветов
    if current_ball['centers']:
        x_coords = np.array([c[0] for c in current_ball['centers']])
        y_coords = np.array([c[1] for c in current_ball['centers']])
        n_points = len(x_coords)
        
        # Создаем градиент цветов от синего (0, 0, 255) до красного (255, 0, 0)
        colors = np.zeros((n_points, 4), dtype=np.uint8)  # RGBA
        for i in range(n_points):
            t = i / max(1, n_points - 1)  # Нормализованный индекс [0, 1]
            colors[i] = [int(255 * t), 0, int(255 * (1 - t)), 255]  # R: 0->255, B: 255->0, A: 255
        
        # Создаем список QBrush для каждой точки
        brushes = [pg.mkBrush(c) for c in colors]
        
        # Обновляем данные с градиентом для символов
        curve_curr_pos.setData(
            x=x_coords, 
            y=y_coords,
            pen='b',  # Линия остается синей для читаемости
            symbol='o', 
            symbolPen='b',  # Контур символов синий
            symbolBrush=brushes  # Список QBrush для градиента символов
        )
        plot_curr_pos.setTitle(f"{labels['curr_pos_title']} ({current_ball['times'][-1]:.2f} s)")

    # График скорости
    if len(current_ball['speeds']) >= 1:
        times = np.array(current_ball['times'])
        speeds_mm_s = np.array(current_ball['speeds']) * config['PIXEL_TO_MM']
        curve_curr_speed.setData(times[1:], speeds_mm_s)

    # График радиуса
    if current_ball['times']:
        curve_curr_radius.setData(current_ball['times'], current_ball['radii'])
        curve_curr_mean_radius.setData(current_ball['times'], current_ball['mean_radii'])

    # Обновление графиков предыдущего шарика и гистограммы
    if previous_ball is not None:
        # Гистограмма
        if ball_avg_radii:
            hist, bins = np.histogram(ball_avg_radii, bins=config['HIST_BINS'])
            plot_dist.clear()
            plot_dist.plot(bins, hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
            plot_dist.setTitle(f"{labels['dist_title']} ({len(ball_avg_radii)} balls)")
            plot_dist.setLabel('bottom', labels['dist_x'])
            plot_dist.setLabel('left', labels['dist_y'])

        # Траектория предыдущего шарика
        prev_x = np.array([c[0] for c in previous_ball['centers']])
        prev_y = np.array([c[1] for c in previous_ball['centers']])
        curve_prev_pos.setData(prev_x, prev_y)
        last_time = previous_ball['times'][-1]
        plot_prev_pos.setTitle(f"{labels['prev_pos_title']} ({last_time:.2f} s)")

        # Скорость предыдущего шарика
        if len(previous_ball['speeds']) >= 1:
            prev_times = np.array(previous_ball['times'])
            prev_speeds_mm_s = np.array(previous_ball['speeds']) * config['PIXEL_TO_MM']
            curve_prev_speed.setData(prev_times[1:], prev_speeds_mm_s)

        # Радиус предыдущего шарика
        curve_prev_radius.setData(previous_ball['times'], previous_ball['radii'])


def execute_sharik_v_unitaze(lang='ru'):
    """
    Основная функция, управляющая процессом обработки видео и построения графиков.
    Видео отображается через OpenCV, графики через PyQtGraph.
    Args:
        lang (str): Language flag ('ru' for Russian, 'en' for English).
    """
    if lang not in ['ru', 'en']:
        raise ValueError("Language must be 'ru' or 'en'")
    
    labels = PLOT_LABELS[lang]
    config = CONFIG
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    # Инициализация OpenCV окон для видео (только два окна: оригинал и морфология)
    window_layout = {
        'Оригинал': (0, 0),  # Верхняя левая часть экрана
        'Морфология': (0, screen_height // 2)  # Нижняя левая часть экрана
    }
    for name, (x, y) in window_layout.items():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        # Размер окон — половина ширины экрана, половина высоты экрана для каждого
        cv2.resizeWindow(name, screen_width // 2, screen_height // 2)
        cv2.moveWindow(name, x, y)

    # Инициализация PyQtGraph для графиков
    app = pg.mkQApp()
    win = pg.GraphicsLayoutWidget(title=labels['window_title'])
    # Правая половина экрана для графиков
    win.resize(screen_width // 2, screen_height)
    win.move(screen_width // 2, 0)  # Перемещаем окно графиков вправо
    win.show()

    # Настройка сетки графиков с явным указанием строк и столбцов
    # 1) Распределение радиусов на всю ширину (строка 0, охват 2 столбцов)
    plot_dist = win.addPlot(row=0, col=0, colspan=2, title=labels['dist_title'])
    plot_dist.setLabel('bottom', labels['dist_x'])
    plot_dist.setLabel('left', labels['dist_y'])

    # 2) Положение текущего шарика (строка 1, столбец 0) — квадратное
    plot_curr_pos = win.addPlot(row=1, col=0, title=labels['curr_pos_title'])
    plot_curr_pos.enableAutoRange(x=False, y=False)
    plot_curr_pos.setXRange(0, 1500, padding=0)
    plot_curr_pos.setAspectLocked(True)
    plot_curr_pos.setLabel('bottom', labels['curr_pos_x'])
    plot_curr_pos.setLabel('left', labels['curr_pos_y'])
    plot_curr_pos.invertY(True)
    curve_curr_pos = plot_curr_pos.plot(
        pen='b', symbol='o', symbolPen='b', symbolBrush='b', name=labels['curr_pos_traj']
    )

    # 3) Положение предыдущего шарика (строка 1, столбец 1) — тоже квадратное
    plot_prev_pos = win.addPlot(row=1, col=1, title=labels['prev_pos_title'])
    plot_prev_pos.setAspectLocked(True)
    plot_prev_pos.enableAutoRange(x=False, y=False)
    plot_prev_pos.setXRange(0, 1500, padding=0)
    plot_prev_pos.setLabel('bottom', labels['prev_pos_x'])
    plot_prev_pos.setLabel('left', labels['prev_pos_y'])
    plot_prev_pos.invertY(True)
    curve_prev_pos = plot_prev_pos.plot(
        pen='y', symbol='o', symbolPen='y', symbolBrush='y', name=labels['prev_pos_traj']
    )

    # 4) Скорость шариков на всю ширину (строка 2, colspan=2)
    plot_speed = win.addPlot(row=2, col=0, colspan=2, title=labels['speed_title'])
    plot_speed.setLabel('bottom', labels['speed_x'])
    plot_speed.setLabel('left', labels['speed_y'])
    curve_curr_speed = plot_speed.plot(
        pen='b', symbol='o', symbolPen='b', symbolBrush='b', name=labels['speed_curr']
    )
    curve_prev_speed = plot_speed.plot(
        pen='y', symbol='o', symbolPen='y', symbolBrush='y', name=labels['speed_prev']
    )

    # 5) Размер шарика на всю ширину (строка 3, colspan=2)
    plot_radius = win.addPlot(row=3, col=0, colspan=2, title=labels['radius_title'])
    plot_radius.setLabel('bottom', labels['radius_x'])
    plot_radius.setLabel('left', labels['radius_y'])
    curve_curr_radius = plot_radius.plot(
        pen='b', symbol='o', symbolPen='b', symbolBrush='b', name=labels['radius_curr']
    )
    curve_curr_mean_radius = plot_radius.plot(
        pen='r', name=labels['radius_mean']
    )
    curve_prev_radius = plot_radius.plot(
        pen='y', symbol='o', symbolPen='y', symbolBrush='y', name=labels['radius_prev']
    )

    # Загрузка видео
    cap = cv2.VideoCapture(config['VIDEO_PATH'])
    if not cap.isOpened():
        print("Ошибка при открытии видеофайла")
        exit()

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        print("Не удалось определить FPS из метаданных видео. Используется значение по умолчанию.")
        frame_rate = config['DEFAULT_FPS']

    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр")
        exit()
    frame_height = frame.shape[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Установка диапазона осей Y после получения frame_height
    plot_curr_pos.setYRange(0, frame_height)
    plot_prev_pos.setYRange(0, frame_height)

    # Инициализация фона для обработки видео
    cropped = frame[:, :1100]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 100, 100)
    background = np.median(filtered).astype(np.float32)
    alpha = 0.1

    # Инициализация данных для шариков
    ball_avg_radii = []
    previous_ball = None
    last_avg_speed = 1.0
    current_ball = {'radii': [], 'centers': [], 'times': [], 'mean_radii': [], 'speeds': []}
    no_ball_frames = 0
    frame_n = 0

    def process_frame():
        nonlocal frame_n, no_ball_frames, current_ball, previous_ball, last_avg_speed, ball_avg_radii, background
        ret, frame = cap.read()
        if not ret:
            app.quit()
            return

        # Обработка видео
        cropped = frame[:, :1100]
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

        # Отображение через OpenCV
        cv2.imshow('Оригинал', cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        cv2.imshow('Морфология', morph)

        # Обработка шариков
        gray_frame = cv2.GaussianBlur(gray, (config['GAUSSIAN_KERNEL_SIZE'], config['GAUSSIAN_KERNEL_SIZE']), config['GAUSSIAN_SIGMA'])
        current_time = frame_n / frame_rate
        ball_data = find_ball(gray_frame, current_ball, current_time, last_avg_speed, config)

        if ball_data is not None:
            ball_x, ball_y, ball_r = ball_data
            optimal_center, optimal_radius = adjust_ball(cropped, gray_frame, ball_x, ball_y, ball_r, frame_height, 1100, config)

            no_ball_frames = 0

            current_ball['radii'].append(optimal_radius)
            current_ball['centers'].append(optimal_center)
            current_ball['times'].append(current_time)

            if len(current_ball['centers']) >= 2:
                positions = np.array(current_ball['centers'])
                times = np.array(current_ball['times'])
                speeds = np.diff(positions, axis=0) / np.diff(times)[:, None]
                speed = np.linalg.norm(speeds[-1])
                current_ball['speeds'].append(speed)
                last_avg_speed = speed

            mean_radius = np.mean(current_ball['radii'])
            current_ball['mean_radii'].append(mean_radius)

            cut_out_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            cv2.circle(cut_out_gray, (ball_x, ball_y), ball_r, (255, 255, 255), config['CIRCLE_THICKNESS'])
            cv2.circle(cut_out_gray, (optimal_center[0], optimal_center[1]), optimal_radius, (127, 127, 127), config['CIRCLE_THICKNESS'])

            update_plots(current_ball, None, ball_avg_radii, None, cut_out_gray, 
                         curve_curr_pos, curve_prev_pos, curve_curr_speed, curve_prev_speed, 
                         curve_curr_radius, curve_curr_mean_radius, curve_prev_radius, 
                         plot_dist, plot_curr_pos, plot_speed, plot_prev_pos, plot_radius, config, lang)

        else:
            no_ball_frames += 1
            if check_ball_status(current_ball, no_ball_frames, config):
                avg_radius = np.mean(current_ball['radii'])
                ball_avg_radii.append(avg_radius)

                if current_ball['speeds']:
                    last_avg_speed = np.mean(current_ball['speeds'])

                previous_ball = current_ball
                current_ball = {'radii': [], 'centers': [], 'times': [], 'mean_radii': [], 'speeds': []}

                update_plots(current_ball, previous_ball, ball_avg_radii, None, gray, 
                             curve_curr_pos, curve_prev_pos, curve_curr_speed, curve_prev_speed, 
                             curve_curr_radius, curve_curr_mean_radius, curve_prev_radius, 
                             plot_dist, plot_curr_pos, plot_speed, plot_prev_pos, plot_radius, config, lang)

                no_ball_frames = 0

        frame_n += 1
        app.processEvents()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            app.quit()

    # Таймер для обработки кадров
    timer = QtCore.QTimer()
    timer.timeout.connect(process_frame)
    timer.start(int(1000 // config['DEFAULT_FPS']))

    # Запуск приложения
    if __name__ == '__main__':
        pg.exec()

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

CONFIG = {
    'VIDEO_PATH': r'D:\Шарыкы\Для машинного зрения\20250506_150457.mp4',
    'DEFAULT_FPS': 60,
    'PIXEL_TO_MM': 22.4 / 356,
    'FIG_WIDTH': 12,
    'FIG_HEIGHT': 10,
    'WINDOW_POSITION_DIVIDER': 4,
    'HOUGH_DP': 1,
    'HOUGH_MIN_DIST': 150,
    'HOUGH_PARAM1': 100,
    'HOUGH_PARAM2': 30,
    'HOUGH_MIN_RADIUS': 40,
    'HOUGH_MAX_RADIUS': 120,
    'MAX_RETRIES': 5,
    'SPEED_THRESHOLD_MULTIPLIER': 3,
    'GAUSSIAN_KERNEL_SIZE': 3,
    'GAUSSIAN_SIGMA': 2,
    'CUT_MASK_OFFSET': 40,
    'VARIATE_STEPS': 1,
    'CENTER_VARIATE_STEPS': 1,
    'CIRCLE_THICKNESS': 2,
    'VIDEO_CLIM_MAX': 255,
    'MAX_NO_BALL_FRAMES': 15,
    'MIN_BALL_DURATION': 4.0,
    'HIST_BINS': 20,
}

if __name__ == "__main__":
    execute_sharik_v_unitaze(lang='en')  # Можно изменить на 'en' для английского
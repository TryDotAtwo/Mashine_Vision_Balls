import cv2
import numpy as np

MainRadius = 22.4

def calculateRadius(rSmall, rBig):
    return rSmall * MainRadius/rBig

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
        circlesBig = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=100, param2=30, minRadius=350, maxRadius=360)
        circlesSmall = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=50, param2=30, minRadius=50, maxRadius=70)
     
        # Подсчет и отображение сферических объектов на кадре   
        if circlesBig is not None and circlesSmall is not None:
            total_objects = len(circlesBig[0])
            circlesBig = np.round(circlesBig[0, :]).astype("int")
            circlesSmall = np.round(circlesSmall[0, :]).astype("int")

            for (x, y, r) in circlesBig:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
                
            for (x1, y1, r1) in circlesSmall:
                cv2.circle(frame, (x1, y1), r1, (255, 0, 0), 3)    
                
            radius = calculateRadius(r1, 355)   
            cv2.putText(frame, f"Diam, mm: {radius}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.8, (0, 0, 255), 4)
            # cv2.putText(frame, f"Diam, mm: {r}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 3.8, (0, 0, 255), 4)
            # cv2.putText(frame, f"cirBig: {circlesBig}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.putText(frame, f"cirSmll: {circlesSmall}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.putText(frame, f"Total Objects: {total_objects}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Подгон 4к к размеру экрана
        k = 60
        frame = cv2.resize(frame, (9*k, 16*k))  
        
        # Отображение текущего кадра с выделенными контурами и информацией о числе обнаруженных объектов
        cv2.imshow("Spherical Objects", frame)

        # Выход из цикла при нажатии клавиши "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

# Путь к видео файлу
video_path = 'D:\Ball12.mp4'

# Подсчет и отображение сферических объектов на видео
count_spherical_objects(video_path)
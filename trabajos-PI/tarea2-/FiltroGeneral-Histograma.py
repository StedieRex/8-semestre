import cv2
import numpy as np
import tkinter as Tk
from tkinter.filedialog import askopenfilename

def cargar_video():
    file_path = askopenfilename(filetypes=[("Video files", "*.mp4")])
    return file_path

def calcular_histograma_hsv(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [2], None, [256], [0, 256])  # Histograma del canal V (iluminación)
    return hist

def aplicar_normalizacion(frame):
    # Normaliza la iluminación del frame usando ecualización del histograma
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_frame[:,:,2] = cv2.equalizeHist(hsv_frame[:,:,2])
    frame_normalizado = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    return frame_normalizado

def promedio_histogramas(hist_list):
    return np.mean(hist_list, axis=0)

def adjust_brightness_contrast(image, alpha=1.5, beta=50):
    # Alpha > 1 aumenta el contraste, Beta > 0 aumenta el brillo
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def camara():
    # Rango para resaltar el color blanco y amarillo en HSV
    h_min_blanco = 0
    h_max_blanco = 180
    s_min_blanco = 0
    s_max_blanco = 60
    v_min_blanco = 200
    v_max_blanco = 255

    h_min_amarillo = 10
    h_max_amarillo = 30
    s_min_amarillo = 100
    s_max_amarillo = 255
    v_min_amarillo = 150
    v_max_amarillo = 255

    # Cargar el archivo de video
    video_path = cargar_video()
    if not video_path:
        print("No se seleccionó ningún archivo de video.")
        return

    # Captura de video
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error al abrir el video")
        exit()

    # Variables para controlar el estado de normalización
    threshold_iluminacion_baja = 0.3  # Umbral para detectar poca luz
    threshold_iluminacion_alta = 0.7  # Umbral para detectar demasiada luz
    intervalo_histograma = 15  # Cada 15 frames
    frame_count = 0
    velocidad = 30  # Ajusta este valor para cambiar la velocidad (menor es más rápido, mayor es más lento)

    # Leer los primeros 5 frames para calcular el histograma promedio de referencia
    hist_list = []
    for i in range(5):
        ret, frame = capture.read()
        if not ret:
            print("Error al leer los primeros frames")
            capture.release()
            exit()
        hist_v = calcular_histograma_hsv(frame)
        hist_list.append(hist_v)

    histograma_referencia = promedio_histogramas(hist_list)
    iluminacion_referencia_total = np.sum(histograma_referencia)

    # Crear una máscara basada en el primer frame
    ret, first_frame = capture.read()
    if not ret:
        print("Error al leer el primer frame")
        capture.release()
        exit()

    mask = np.zeros(first_frame.shape[:2], dtype="uint8")
    puntos = np.array([[481, 466], [742, 457], [1206, 681], [153, 697]])
    cv2.fillPoly(mask, [puntos], 255)

    aplicar_filtro = False

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame_count += 1

        # Cada 15 frames, calcular el histograma HSV y compararlo con el de referencia
        if frame_count % intervalo_histograma == 0:
            hist_v = calcular_histograma_hsv(frame)
            iluminacion_actual_total = np.sum(hist_v)
            iluminacion_baja_val = np.sum(hist_v[:50]) / iluminacion_actual_total  # Proporción de píxeles con poca luz
            iluminacion_alta_val = np.sum(hist_v[200:]) / iluminacion_actual_total  # Proporción de píxeles con demasiada luz
            
            diferencia_iluminacion = iluminacion_actual_total / iluminacion_referencia_total

            # Detectar si la iluminación es demasiado baja o alta en comparación con el promedio de referencia
            if (iluminacion_baja_val > threshold_iluminacion_baja or
                iluminacion_alta_val > threshold_iluminacion_alta or
                diferencia_iluminacion < 0.55 or diferencia_iluminacion > 1):
                aplicar_filtro = True
            else:
                aplicar_filtro = False

        # Aplicar la máscara al frame original
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Si la iluminación es inadecuada (baja o alta), normalizar el frame
        if aplicar_filtro:
            masked_frame = aplicar_normalizacion(masked_frame)

            # Resaltar blanco y amarillo en el frame normalizado
            hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
            mascara_blanco = cv2.inRange(hsv, (h_min_blanco, s_min_blanco, v_min_blanco), (h_max_blanco, s_max_blanco, v_max_blanco))
            mascara_amarillo = cv2.inRange(hsv, (h_min_amarillo, s_min_amarillo, v_min_amarillo), (h_max_amarillo, s_max_amarillo, v_max_amarillo))
            mascara = cv2.bitwise_or(mascara_blanco, mascara_amarillo)
            masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mascara)

        else:
            # Resaltar blanco y amarillo en el frame original
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mascara_blanco = cv2.inRange(hsv, (h_min_blanco, s_min_blanco, v_min_blanco), (h_max_blanco, s_max_blanco, v_max_blanco))
            mascara_amarillo = cv2.inRange(hsv, (h_min_amarillo, s_min_amarillo, v_min_amarillo), (h_max_amarillo, s_max_amarillo, v_max_amarillo))
            mascara = cv2.bitwise_or(mascara_blanco, mascara_amarillo)
            result = cv2.bitwise_and(frame, frame, mask=mascara)

            # Ajustar brillo y contraste en las áreas resaltadas
            masked_frame = adjust_brightness_contrast(result)

        # Mostrar el resultado
        cv2.imshow('Masked Video', masked_frame)
        if cv2.waitKey(velocidad) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camara()
    print("Fin del programa")

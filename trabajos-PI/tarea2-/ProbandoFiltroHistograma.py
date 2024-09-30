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

# Cargar el archivo de video
video_path = cargar_video()
if not video_path:
    print("No se seleccionó ningún archivo de video.")
    exit()

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

# Variable para controlar la velocidad de reproducción (en milisegundos)
velocidad = 30  # Ajusta este valor para cambiar la velocidad (menor es más rápido, mayor es más lento)

# Leer los primeros 5 frames para calcular el histograma promedio de referencia
hist_list = []
for i in range(5):
    ret, frame = capture.read()
    if not ret:
        print("Error al leer los primeros frames")
        capture.release()
        exit()
    
    # Calcular histograma del canal V de cada frame y añadirlo a la lista
    hist_v = calcular_histograma_hsv(frame)
    hist_list.append(hist_v)

# Promediar los histogramas de los primeros 5 frames para obtener el histograma de referencia
histograma_referencia = promedio_histogramas(hist_list)
iluminacion_referencia_total = np.sum(histograma_referencia)

# Crear una máscara basada en el primer frame
mask = np.zeros(frame.shape[:2], dtype="uint8")
puntos = np.array([[481, 466], [742, 457], [1206, 681], [153, 697]])
cv2.fillPoly(mask, [puntos], 255)

aplicar_filtro = False
# Procesar los frames restantes
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
        
        # Detectar si la iluminación es demasiado baja o alta en comparación con el promedio de referencia
        diferencia_iluminacion = iluminacion_actual_total / iluminacion_referencia_total

        # Si hay poca luz o demasiada luz, aplicar normalización
        # El 0.8 y 1.2 son valores arbitrarios, puedes ajustarlos según tus necesidades
        # 0.8 significa que la iluminación actual es un 20% menor que la de referencia
        # 1.2 significa que la iluminación actual es un 20% mayor que la de referencia, puede tomar valores entre 0 y 1
        if iluminacion_baja_val > threshold_iluminacion_baja or iluminacion_alta_val > threshold_iluminacion_alta or diferencia_iluminacion < 0.55 or diferencia_iluminacion > 1:
            aplicar_filtro = True
        else:
            aplicar_filtro = False

    # Aplicar la máscara al frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Si la iluminación es inadecuada (baja o alta), normalizar el frame
    if aplicar_filtro:
        masked_frame = aplicar_normalizacion(masked_frame)

    # Mostrar el resultado
    cv2.imshow('Masked Video', masked_frame)
    if cv2.waitKey(velocidad) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

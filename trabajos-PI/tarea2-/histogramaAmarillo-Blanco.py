import cv2
import numpy as np
import tkinter as Tk
from tkinter.filedialog import askopenfilename

def cargar_video():
    file_path = askopenfilename(filetypes=[("Video files", "*.mp4")])
    return file_path

def calcular_histograma_hsv(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Histograma por canales (H, S, V)
    hist_h = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])  # Canal H (0-180 para OpenCV)
    hist_s = cv2.calcHist([hsv_frame], [1], None, [256], [0, 256])  # Canal S
    hist_v = cv2.calcHist([hsv_frame], [2], None, [256], [0, 256])  # Canal V
    return hist_h, hist_s, hist_v

def aplicar_normalizacion_especifica(frame, blanco_referencia, amarillo_referencia):
    # Convertir el frame a HSV para trabajar en esos canales
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Ecualizar el canal V para mejorar la iluminación global
    hsv_frame[:, :, 2] = cv2.equalizeHist(hsv_frame[:, :, 2])

    # Aplicar la ecualización y ajustes personalizados para destacar el blanco y amarillo
    s_channel = hsv_frame[:, :, 1]
    v_channel = hsv_frame[:, :, 2]

    # Resaltar amarillo (Hue alrededor de 20-30°)
    amarillo_mask = (hsv_frame[:, :, 0] >= 20) & (hsv_frame[:, :, 0] <= 30)
    
    # Aplica la ecualización en la región de la máscara amarilla
    s_channel[amarillo_mask] = np.clip(cv2.equalizeHist(s_channel[amarillo_mask].astype(np.uint8)), 0, 255)
    v_channel[amarillo_mask] = np.clip(cv2.equalizeHist(v_channel[amarillo_mask].astype(np.uint8)), 0, 255)

    # Resaltar blanco (Alta V, baja S)
    blanco_mask = (v_channel >= 200) & (s_channel <= 50)
    
    # Aplica la ecualización en la región de la máscara blanca
    s_channel[blanco_mask] = np.clip(cv2.equalizeHist(s_channel[blanco_mask].astype(np.uint8)), 0, 255)
    v_channel[blanco_mask] = np.clip(cv2.equalizeHist(v_channel[blanco_mask].astype(np.uint8)), 0, 255)

    # Actualizar el frame con los canales ajustados
    hsv_frame[:, :, 1] = s_channel
    hsv_frame[:, :, 2] = v_channel

    # Convertir el frame de regreso a BGR para mostrarlo
    frame_mejorado = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    return frame_mejorado

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
intervalo_histograma = 15  # Cada 15 frames
frame_count = 0

# Variable para controlar la velocidad de reproducción (en milisegundos)
velocidad = 30  # Ajusta este valor para cambiar la velocidad (menor es más rápido, mayor es más lento)

# Leer los primeros 5 frames para calcular el histograma promedio de referencia
hist_list_h = []
hist_list_s = []
hist_list_v = []
for i in range(5):
    ret, frame = capture.read()
    if not ret:
        print("Error al leer los primeros frames")
        capture.release()
        exit()
    
    # Calcular histograma de cada canal y añadirlo a las listas
    hist_h, hist_s, hist_v = calcular_histograma_hsv(frame)
    hist_list_h.append(hist_h)
    hist_list_s.append(hist_s)
    hist_list_v.append(hist_v)

# Promediar los histogramas de los primeros 5 frames para obtener el histograma de referencia
histograma_referencia_h = promedio_histogramas(hist_list_h)
histograma_referencia_s = promedio_histogramas(hist_list_s)
histograma_referencia_v = promedio_histogramas(hist_list_v)

# Crear una máscara basada en el primer frame
mask = np.zeros(frame.shape[:2], dtype="uint8")
puntos = np.array([[481, 466], [742, 457], [1206, 681], [153, 697]])
cv2.fillPoly(mask, [puntos], 255)

# Procesar los frames restantes
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    frame_count += 1

    # Cada 15 frames, aplicar la normalización
    if frame_count % intervalo_histograma == 0:
        # Mejorar la iluminación de los colores blanco y amarillo
        frame_mejorado = aplicar_normalizacion_especifica(frame, histograma_referencia_s, histograma_referencia_v)
    else:
        frame_mejorado = frame

    # Aplicar la máscara al frame
    masked_frame = cv2.bitwise_and(frame_mejorado, frame_mejorado, mask=mask)

    # Mostrar el resultado
    cv2.imshow('Masked Video', masked_frame)
    if cv2.waitKey(velocidad) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
import cv2
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np

import cv2

def cargar_video():
    Tk().withdraw()  # Ocultar la ventana principal de tkinter
    file_path = askopenfilename(filetypes=[("Video files", "*.mp4")])
    return file_path

# Cargar el archivo de video
video_path = cargar_video()
if not video_path:
    print("No se seleccionó ningún archivo de video.")
    exit()
    
cap = cv2.VideoCapture(video_path)
# Comprobar si el video se abrió correctamente
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

# Obtener el ancho, alto y el frame rate del video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Definir el codec y crear el objeto de escritura de video
out = cv2.VideoWriter('video_negativo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Procesar el video frame por frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar el negativo
    negative_frame = 255 - frame

    # Mostrar el frame negativo
    cv2.imshow('Video Negativo', negative_frame)

    # Guardar el frame negativo en el archivo de salida
    out.write(negative_frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

# Liberar el objeto del video y cerrar todas las ventanas
cap.release()
out.release()
cv2.destroyAllWindows()


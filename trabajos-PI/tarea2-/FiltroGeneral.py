import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def apply_gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_brightness_contrast(image, alpha=1.5, beta=50):
    # Alpha > 1 aumenta el contraste, Beta > 0 aumenta el brillo
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def create_mask(image, lower_bound, upper_bound):
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return mask

def highlight_yellow_and_white(frame):
    # Convertir a espacio de color HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definir rangos para color amarillo en HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Definir rangos para color blanco en HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    
    # Crear máscaras para amarillo y blanco
    yellow_mask = create_mask(hsv_frame, lower_yellow, upper_yellow)
    white_mask = create_mask(hsv_frame, lower_white, upper_white)
    
    # Combinar las máscaras
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    
    # Aplicar la máscara al frame original
    masked_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)
    
    # Ajustar brillo y contraste solo en las áreas resaltadas (amarillo y blanco)
    highlighted = adjust_brightness_contrast(masked_frame)
    
    # Combinar el resultado resaltado con el frame original
    final_frame = cv2.addWeighted(frame, 1.0, highlighted, 1.0, 0)
    
    return final_frame

def process_video():
    # Crear una ventana de diálogo para seleccionar el archivo de video
    Tk().withdraw()  # Ocultar la ventana principal de tkinter
    input_video_path = askopenfilename(title="Seleccionar un video", filetypes=[("Archivos de video", "*.mp4 *.avi")])
    
    if not input_video_path:
        print("No se seleccionó ningún archivo")
        return
    
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error al abrir el video")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resaltar los colores amarillo y blanco
        final_frame = highlight_yellow_and_white(frame)
        
        # Mostrar el video procesado en una ventana
        cv2.imshow('Video Procesado', final_frame)
        
        # Presionar 'q' para salir del video
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función para procesar el video
process_video()

import cv2
import numpy as np
import tkinter as Tk
from tkinter.filedialog import askopenfilename

def apply_gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_brightness_contrast(image, alpha=0.75, beta=45):
    # Alpha > 1 aumenta el contraste, Beta > 0 aumenta el brillo
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def create_mask(image, lower_bound, upper_bound):
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return mask

def highlight_yellow_and_white(frame, gamma):
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
    
    # Aplicar corrección de gamma para reducir sombras
    gamma_corrected_frame = apply_gamma_correction(masked_frame, gamma=gamma)
    
    # Ajustar brillo y contraste solo en las áreas resaltadas (amarillo y blanco)
    highlighted = adjust_brightness_contrast(gamma_corrected_frame)
    
    # Combinar el resultado resaltado con el frame original
    final_frame = cv2.addWeighted(frame, 1.0, highlighted, 1.0, 0)
    
    return final_frame

def cargar_video():
    file_path = askopenfilename(filetypes=[("Video files", "*.mp4")])
    return file_path

def calcular_histograma_hsv(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [2], None, [256], [0, 256])  # Histograma del canal V (iluminación)
    return hist

def promedio_histogramas(hist_list):
    return np.mean(hist_list, axis=0)

def adjust_brightness_contrast(image, alpha=1.5, beta=50):
    # Alpha > 1 aumenta el contraste, Beta > 0 aumenta el brillo
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def camara():
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

    threshold_iluminacion_baja = 0.2
    intervalo_histograma = 5
    frame_count = 0
    velocidad = 5
    nivelGamma = 15.0/10.0

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

        if frame_count % intervalo_histograma == 0:
            hist_v = calcular_histograma_hsv(frame)
            iluminacion_actual_total = np.sum(hist_v)
            iluminacion_baja_val = np.sum(hist_v[:50]) / iluminacion_actual_total
            diferencia_iluminacion = iluminacion_actual_total / iluminacion_referencia_total

            if (iluminacion_baja_val > threshold_iluminacion_baja or diferencia_iluminacion < 0.55 ): # si la iluminacion es menor a 55% significa que esta muy oscuro
                aplicar_filtro = True
            else:
                aplicar_filtro = False

        #masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        #aplicar_filtro = False
        
        if aplicar_filtro:
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            masked_frame = highlight_yellow_and_white(masked_frame, nivelGamma)
            
            hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
            mascara_blanco = cv2.inRange(hsv, (h_min_blanco, s_min_blanco, v_min_blanco), (h_max_blanco, s_max_blanco, v_max_blanco))
            mascara_amarillo = cv2.inRange(hsv, (h_min_amarillo, s_min_amarillo, v_min_amarillo), (h_max_amarillo, s_max_amarillo, v_max_amarillo))
            mascara = cv2.bitwise_or(mascara_blanco, mascara_amarillo)
            masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mascara)
            
        else:
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            #entre mas alto beta mas brillo, sus limites son 0 y 100, puede ser negativo
            #entre mas alto alpha mas contraste, sus limites son 0 y 3, puede ser negativo
            masked_frame = adjust_brightness_contrast(masked_frame, alpha=0.75, beta=45)
            
            hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
            mascara_blanco = cv2.inRange(hsv, (h_min_blanco, s_min_blanco, v_min_blanco), (h_max_blanco, s_max_blanco, v_max_blanco))
            mascara_amarillo = cv2.inRange(hsv, (h_min_amarillo, s_min_amarillo, v_min_amarillo), (h_max_amarillo, s_max_amarillo, v_max_amarillo))
            mascara = cv2.bitwise_or(mascara_blanco, mascara_amarillo)
            masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mascara)
            
            
        cv2.imshow('Masked Video', masked_frame)
        if cv2.waitKey(velocidad) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camara()
    print("Fin del programa")

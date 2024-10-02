import cv2
import numpy as np
import tkinter as Tk
from tkinter.filedialog import askopenfilename

def compute_histogram(image_channel, bins=256):
    hist, _ = np.histogram(image_channel, bins=bins, range=(0, bins))
    return hist

def compute_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf.max())
    return cdf_normalized

def histogram_specification(original_hist, reference_points, bins=256):
    P_A = compute_cdf(original_hist)
    K = bins
    f_hs = np.zeros(K, dtype=np.float32)

    for a in range(K):
        b = P_A[a]
        if b <= reference_points[0][1]:
            a_prime = reference_points[0][0]
        elif b >= reference_points[-1][1]:
            a_prime = reference_points[-1][0]
        else:
            n = len(reference_points) - 1
            while n >= 0 and b < reference_points[n][1]:
                n -= 1
            n = max(n, 0)
            a_prime = reference_points[n][0] + (b - reference_points[n][1]) * \
                      ((reference_points[n + 1][0] - reference_points[n][0]) / 
                      (reference_points[n + 1][1] - reference_points[n][1]))
        f_hs[a] = np.clip(a_prime, 0, 255)

    return f_hs

def apply_histogram_specification(image, reference_points, bins=256):
    channels = cv2.split(image)
    result_channels = []

    for channel in channels:
        original_hist = compute_histogram(channel, bins=bins)
        f_hs = histogram_specification(original_hist, reference_points, bins=bins)
        
        # Remapear los valores del canal original usando f_hs
        new_channel = np.clip(np.interp(channel.flatten(), np.arange(bins), f_hs), 0, 255).reshape(channel.shape)
        result_channels.append(new_channel.astype(np.uint8))
    
    result_image = cv2.merge(result_channels)
    return result_image

def cargar_video():
    file_path = askopenfilename(filetypes=[("Video files", "*.mp4")])
    return file_path

def calcular_histograma_hsv(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [2], None, [256], [0, 256])  # Histograma del canal V (iluminación)
    return hist

def aplicar_normalizacion(frame):
    # Puntos de referencia para la especificación del histograma
    reference_points = [
        (0, 0.0),
        (20, 0.23),  
        (89, 0.66),  
        (96, 0.77),  
        (148, 0.86),  
        (180, 0.79),  
        (237, 0.84),  
        (255, 0.97)
    ]
    
    # Aplicar la especificación del histograma en el frame
    frame_normalizado = apply_histogram_specification(frame, reference_points)
    return frame_normalizado

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

    threshold_iluminacion_baja = 0.3
    threshold_iluminacion_alta = 0.7
    intervalo_histograma = 15
    frame_count = 0
    velocidad = 5

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
            iluminacion_alta_val = np.sum(hist_v[200:]) / iluminacion_actual_total
            diferencia_iluminacion = iluminacion_actual_total / iluminacion_referencia_total

            if (iluminacion_baja_val > threshold_iluminacion_baja or
                iluminacion_alta_val > threshold_iluminacion_alta or
                diferencia_iluminacion < 0.55 or diferencia_iluminacion > 1):
                aplicar_filtro = True
            else:
                aplicar_filtro = False

        #masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        if aplicar_filtro:
            frame = aplicar_normalizacion(frame)
            masked_frame = frame
            # masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
            # mascara_blanco = cv2.inRange(hsv, (h_min_blanco, s_min_blanco, v_min_blanco), (h_max_blanco, s_max_blanco, v_max_blanco))
            # mascara_amarillo = cv2.inRange(hsv, (h_min_amarillo, s_min_amarillo, v_min_amarillo), (h_max_amarillo, s_max_amarillo, v_max_amarillo))
            # mascara = cv2.bitwise_or(mascara_blanco, mascara_amarillo)
            # masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mascara)
            
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

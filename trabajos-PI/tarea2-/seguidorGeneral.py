import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def calcular_histograma_hsv(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [2], None, [256], [0, 256])  # Histograma del canal V (iluminación)
    return hist

def promedio_histogramas(hist_list):
    return np.mean(hist_list, axis=0)

def ajustar_gamma(imagen, gamma=1.0):
    # Crear una tabla de corrección gamma
    tabla = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(imagen, tabla)

def adjust_brightness_contrast(image, alpha, beta):
# Convertir la imagen a un array de NumPy
    img_array = np.array(image, dtype=np.uint8)
    
    # Aplicar la transformación de contraste a cada canal (R, G, B)
    img_contraste = np.zeros_like(img_array)
    for i in range(3):  # Para cada canal de color
        img_contraste[..., i] = np.clip(alpha * img_array[..., i] + beta, 0, 255)
    
    return img_contraste.astype(np.uint8)

def camara():
    # Rango para resaltar el color blanco y amarillo en HSV
    # Blanco: Baja saturación, alto brillo
    h_min_blanco = 0
    h_max_blanco = 180  # Hue en OpenCV va de 0 a 180
    s_min_blanco = 0
    s_max_blanco = 60  # Saturación baja para captar el blanco
    v_min_blanco = 200
    v_max_blanco = 255  # Valores altos de brillo

    # # Rango para amarillo
    # h_min_amarillo = 0
    # h_max_amarillo = 34  # Tono específico del amarillo
    # s_min_amarillo = 82
    # s_max_amarillo = 230  # Saturación más alta para amarillo
    # v_min_amarillo = 130
    # v_max_amarillo = 255  # Amarillo es brillante también
    
    # Rango para amarillo
    h_min_amarillo = 0
    h_max_amarillo = 100  # Tono específico del amarillo
    s_min_amarillo = 100
    s_max_amarillo = 255  # Saturación más alta para amarillo
    v_min_amarillo = 120
    v_max_amarillo = 255  # Amarillo es brillante también

    def cargar_video():
        Tk().withdraw()  # Ocultar la ventana principal de tkinter
        file_path = askopenfilename(filetypes=[("Video files", "*.mp4")])
        return file_path

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

    # Mascara para deliminar la carretera
    # Leer el primer frame para usarlo como referencia
    ret, first_frame = capture.read()
    if not ret:
        print("Error al leer el primer frame")
        capture.release()
        exit()

    # Crear una máscara basada en el primer frame
    mask = np.zeros(first_frame.shape[:2], dtype="uint8")

    # Define las coordenadas de las cuatro esquinas del rectángulo irregular
    puntos = np.array([[481, 466], [742, 457], [1206, 681], [153, 697]])

    # Rellenar el polígono (rectángulo irregular) en la máscara
    cv2.fillPoly(mask, [puntos], 255)

    # Ajustar brillo y contraste
    gamma = 56.0/ 100.0 + 0.1  # Rango de gamma ajustado a 0.1 - 1.1
    alpha = 88.0 / 50.0  # Rango de contraste ajustado a 0.0 - 2.0
    beta = 1.0 # Rango de contraste ajustado a 0.0 - 2.0

    # Leer los primeros 5 frames para calcular el histograma promedio de referencia
    hist_list = []
    for i in range(5):
        ret, frame = capture.read()
        mascara_delimitadora = cv2.bitwise_and(frame, frame, mask=mask)
        ajuste_contraste = adjust_brightness_contrast(mascara_delimitadora, alpha, beta)

        if not ret:
            print("Error al leer los primeros frames")
            capture.release()
            exit()
        
        # Calcular histograma del canal V de cada frame y añadirlo a la lista
        hist_v = calcular_histograma_hsv(mascara_delimitadora)
        hist_list.append(hist_v)
    
    frame_count = 0
    intervalo_histograma = 1  # Cada 15 frames
    # Variables para controlar el estado de normalización
    threshold_iluminacion_baja = 0.3  # Umbral para detectar poca luz, entre mas bajo mas oscuro
    threshold_iluminacion_alta = 0.18  # Umbral para detectar demasiada luz, entre mas alto mas oscuro

    # Promediar los histogramas de los primeros 5 frames para obtener el histograma de referencia
    histograma_referencia = promedio_histogramas(hist_list)
    iluminacion_referencia_total = np.sum(histograma_referencia)
    print("Histograma de referencia: ", iluminacion_referencia_total)
    
    aplicar_filtro = False
    
    while capture.isOpened():
        ret, frame = capture.read()
        mascara_delimitadora = cv2.bitwise_and(frame, frame, mask=mask)

        if ret:
            frame_count += 1
            
            if not aplicar_filtro:
                
                # Ajustar brillo y contraste en las áreas resaltadas
                ajuste_contraste = adjust_brightness_contrast(mascara_delimitadora, alpha, beta)
                ajuste_fin = ajustar_gamma(ajuste_contraste, gamma)
                
                # Convertir a espacio de color HSV
                hsv = cv2.cvtColor(ajuste_fin, cv2.COLOR_BGR2HSV)

                # Aplicar los valores HSV para resaltar el color blanco
                mascara_blanco = cv2.inRange(hsv, (h_min_blanco, s_min_blanco, v_min_blanco), (h_max_blanco, s_max_blanco, v_max_blanco))

                # Aplicar los valores HSV para resaltar el color amarillo
                mascara_amarillo = cv2.inRange(hsv, (h_min_amarillo, s_min_amarillo, v_min_amarillo), (h_max_amarillo, s_max_amarillo, v_max_amarillo))

                # Combinar ambas máscaras (blanco y amarillo)
                mascara = cv2.bitwise_or(mascara_blanco, mascara_amarillo)
                frame_final = cv2.bitwise_and(ajuste_fin, ajuste_fin, mask=mascara)
            else:                
                # Convertir a espacio de color HSV
                hsv = cv2.cvtColor(mascara_delimitadora, cv2.COLOR_BGR2HSV)

                # Aplicar los valores HSV para resaltar el color blanco
                mascara_blanco = cv2.inRange(hsv, (h_min_blanco, s_min_blanco, v_min_blanco), (h_max_blanco, s_max_blanco, v_max_blanco))

                # Aplicar los valores HSV para resaltar el color amarillo
                mascara_amarillo = cv2.inRange(hsv, (h_min_amarillo, s_min_amarillo, v_min_amarillo), (h_max_amarillo, s_max_amarillo, v_max_amarillo))

                # Combinar ambas máscaras (blanco y amarillo)
                mascara = cv2.bitwise_or(mascara_blanco, mascara_amarillo)
                frame_final = cv2.bitwise_and(mascara_delimitadora, mascara_delimitadora, mask=mascara)                

            # Cada 15 frames, calcular el histograma HSV y compararlo con el de referencia
            if frame_count % intervalo_histograma == 0:
                    hist_v = calcular_histograma_hsv(frame_final)
                    iluminacion_actual_total = np.sum(hist_v)
                    iluminacion_alta_val = np.sum(hist_v[200:]) / iluminacion_actual_total  # Proporción de píxeles con demasiada luz
                    
                    # Detectar si la iluminación es demasiado baja o alta en comparación con el promedio de referencia
                    diferencia_iluminacion = iluminacion_actual_total / iluminacion_referencia_total
                    print("valores de iluminacion alta: ", (iluminacion_alta_val*100))
                    #comparacion para iluminacion alto
                    if iluminacion_alta_val > threshold_iluminacion_alta:
                        aplicar_filtro = True
                        #print("Iluminación alta")
                    else:
                        aplicar_filtro = False
                        #print("Iluminación normal")
            # Mostrar la imagen enmascarada y el video original
            cv2.imshow('Original', frame)
            cv2.imshow('Resaltado Blanco y Amarillo', frame_final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camara()
    print("Fin del programa")

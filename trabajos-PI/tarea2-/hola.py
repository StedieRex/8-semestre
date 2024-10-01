import cv2
import numpy as np
from tkinter import Tk
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
        f_hs[a] = np.clip(a_prime, 0, 255)  # Asegúrate de que el nuevo valor esté en el rango correcto

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

def camara():
    # Rango para resaltar el color blanco y amarillo en HSV
    h_min_blanco = 0
    h_max_blanco = 180  
    s_min_blanco = 0
    s_max_blanco = 60  
    v_min_blanco = 200
    v_max_blanco = 255  

    # Rango para amarillo
    h_min_amarillo = 10
    h_max_amarillo = 30  # Tono específico del amarillo
    s_min_amarillo = 100
    s_max_amarillo = 255  
    v_min_amarillo = 150
    v_max_amarillo = 255  

    def cargar_video():
        Tk().withdraw()  
        file_path = askopenfilename(filetypes=[("Video files", "*.mp4")])
        return file_path

    video_path = cargar_video()
    if not video_path:
        print("No se seleccionó ningún archivo de video.")
        return

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error al abrir el video")
        exit()

    # Leer el primer frame para usarlo como referencia
    ret, first_frame = capture.read()
    if not ret:
        print("Error al leer el primer frame")
        capture.release()
        exit()

    # Especificación del histograma del primer frame
    reference_points = [
        (0, 0.0),
        (32, 0.06), 
        (64, 0.51),
        (96, 0.55),
        (128, 0.50),
        (164, 0.67),
        (181, 0.49),
        (220, 0.98),
        (255, 1)
    ]

    # Aplicar la especificación del histograma al primer frame
    first_frame_adjusted = apply_histogram_specification(first_frame, reference_points)

    # Crear una máscara basada en el primer frame ajustado
    mask = np.zeros(first_frame_adjusted.shape[:2], dtype="uint8")
    puntos = np.array([[481, 466], [742, 457], [1206, 681], [153, 697]])
    cv2.fillPoly(mask, [puntos], 255)

    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            # Aplicar la especificación del histograma a cada frame
            frame_adjusted = apply_histogram_specification(frame, reference_points)

            # Aplicar la máscara delimitadora de la carretera
            mascara_delimitadora = cv2.bitwise_and(frame_adjusted, frame_adjusted, mask=mask)
            frame_hsv = cv2.cvtColor(mascara_delimitadora, cv2.COLOR_BGR2HSV)

            # Aplicar las máscaras para resaltar colores
            mascara_blanco = cv2.inRange(frame_hsv, (h_min_blanco, s_min_blanco, v_min_blanco), (h_max_blanco, s_max_blanco, v_max_blanco))
            mascara_amarillo = cv2.inRange(frame_hsv, (h_min_amarillo, s_min_amarillo, v_min_amarillo), (h_max_amarillo, s_max_amarillo, v_max_amarillo))
            mascara = cv2.bitwise_or(mascara_blanco, mascara_amarillo)
            result = cv2.bitwise_and(frame_hsv, frame_hsv, mask=mascara)

            # Mostrar la imagen enmascarada y el video original
            cv2.imshow('Original', frame)
            cv2.imshow('Resaltado Blanco y Amarillo', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camara()
    print("Fin del programa")

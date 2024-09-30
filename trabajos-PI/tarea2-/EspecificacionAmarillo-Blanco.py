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

def select_image():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    root.destroy()
    return file_path

image_path = select_image()

if not image_path:
    raise FileNotFoundError("No se seleccionó ninguna imagen")

# Cargar la imagen en BGR (OpenCV lo carga en BGR por defecto)
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("La imagen no pudo ser cargada")

# Convertir BGR a RGB para la visualización
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Definir puntos de control de la distribución de referencia (L_R) con el doble de puntos
reference_points = [
    (0, 0.0),
    (32, 0.10),  
    (64, 0.20),  
    (96, 0.30),  
    (128, 0.50),  
    (160, 0.70),  
    (192, 0.85),  
    (220, 0.95),  
    (255, 1.0)
]

# Aplicar la especificación del histograma
result_image = apply_histogram_specification(img_rgb, reference_points)

# Mostrar ambas imágenes en una ventana
combined_image = np.hstack((img_rgb, result_image))
cv2.imshow('Imagen Original y Procesada', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

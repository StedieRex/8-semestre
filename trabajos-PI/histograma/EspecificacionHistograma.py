import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def compute_histogram(image_channel, bins=256):
    # Calcula el histograma del canal de la imagen
    hist, _ = np.histogram(image_channel, bins=bins, range=(0, bins))
    return hist

def compute_cdf(hist):
    # Calcula la función de distribución acumulativa (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf.max())  # Normaliza el CDF
    return cdf_normalized

def histogram_specification(original_hist, reference_points, bins=256):
    P_A = compute_cdf(original_hist)
    K = bins
    
    # Crear tabla de mapeo f_hs
    f_hs = np.zeros(K, dtype=np.float32)
    
    for a in range(K - 1):
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
            # Interpolación lineal entre los puntos de control
            a_prime = reference_points[n][0] + (b - reference_points[n][1]) * \
                    ((reference_points[n + 1][0] - reference_points[n][0]) / 
                    (reference_points[n + 1][1] - reference_points[n][1]))
        f_hs[a] = a_prime

    return f_hs

def apply_histogram_specification(image, reference_points, bins=256):
    # Separar la imagen en canales R, G, B
    channels = cv2.split(image)
    result_channels = []

    # Aplicar la especificación del histograma a cada canal
    for channel in channels:
        original_hist = compute_histogram(channel, bins=bins)
        f_hs = histogram_specification(original_hist, reference_points, bins=bins)

        # Remapear los valores del canal original usando f_hs
        new_channel = np.clip(np.interp(channel.flatten(), np.arange(bins), f_hs), 0, 255).reshape(channel.shape)
        result_channels.append(new_channel.astype(np.uint8))
    
    # Combinar los canales nuevamente
    result_image = cv2.merge(result_channels)
    return result_image

# Ejemplo de uso:
# Cargar una imagen
# Abrir la ventana de diálogo para seleccionar la imagen

def select_image():
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
    file_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    root.destroy()
    return file_path

image_path = select_image()

if not image_path:
    raise FileNotFoundError("No se seleccionó ninguna imagen")

# Cargar la imagen seleccionada en color
img = cv2.imread(image_path)  # Cargar la imagen en BGR
if img is None:
    raise FileNotFoundError("La imagen no pudo ser cargada")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir BGR a RGB

# Definir puntos de control de la distribución de referencia (L_R)
reference_points = [
    (0, 0.0),
    (64, 0.25),
    (128, 0.5),
    (192, 0.75),
    (255, 1.0)
]

# Aplicar la especificación del histograma
result_image = apply_histogram_specification(img, reference_points)

# Guardar la imagen resultante
cv2.imwrite('result_image.png', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))  # Guardar en BGR

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
        f_hs[a] = np.clip(a_prime, 0, 255)

    return f_hs

def apply_histogram_specification(image, reference_points, bins=256):
    channels = cv2.split(image)
    result_channels = []

    for channel in channels:
        original_hist = compute_histogram(channel, bins=bins)
        f_hs = histogram_specification(original_hist, reference_points, bins=bins)
        
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

def update_histogram(val):
    reference_points = [
        (0, 0.0),
        (32, cv2.getTrackbarPos('P1', 'Ajuste de Histograma') / 100),
        (64, cv2.getTrackbarPos('P2', 'Ajuste de Histograma') / 100),
        (96, cv2.getTrackbarPos('P3', 'Ajuste de Histograma') / 100),
        (128, cv2.getTrackbarPos('P4', 'Ajuste de Histograma') / 100),
        (160, cv2.getTrackbarPos('P5', 'Ajuste de Histograma') / 100),
        (192, cv2.getTrackbarPos('P6', 'Ajuste de Histograma') / 100),
        (220, cv2.getTrackbarPos('P7', 'Ajuste de Histograma') / 100),
        (255, cv2.getTrackbarPos('P8', 'Ajuste de Histograma') / 100)
    ]
    
    adjusted_image = apply_histogram_specification(img, reference_points)
    
    combined_image = np.hstack((img, adjusted_image))
    cv2.imshow('Ajuste de Histograma', combined_image)

# Cargar la imagen
image_path = select_image()
if not image_path:
    raise FileNotFoundError("No se seleccionó ninguna imagen")

img = cv2.imread(image_path) 
if img is None:
    raise FileNotFoundError("La imagen no pudo ser cargada")

# Crear una ventana para los sliders
cv2.namedWindow('Ajuste de Histograma', cv2.WINDOW_NORMAL)  # Permitir el cambio de tamaño de la ventana

# Crear sliders más pequeños
cv2.createTrackbar('P1', 'Ajuste de Histograma', 10, 50, update_histogram)
cv2.createTrackbar('P2', 'Ajuste de Histograma', 20, 50, update_histogram)
cv2.createTrackbar('P3', 'Ajuste de Histograma', 30, 50, update_histogram)
cv2.createTrackbar('P4', 'Ajuste de Histograma', 40, 50, update_histogram)
cv2.createTrackbar('P5', 'Ajuste de Histograma', 50, 50, update_histogram)
cv2.createTrackbar('P6', 'Ajuste de Histograma', 60, 50, update_histogram)
cv2.createTrackbar('P7', 'Ajuste de Histograma', 70, 50, update_histogram)
cv2.createTrackbar('P8', 'Ajuste de Histograma', 80, 50, update_histogram)

# Llamar una vez para inicializar la visualización
update_histogram(0)

# Mantener la ventana abierta hasta que se presione la tecla 'q'
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

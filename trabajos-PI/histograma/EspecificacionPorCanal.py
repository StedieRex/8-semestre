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

    for i, channel in enumerate(channels):
        original_hist = compute_histogram(channel, bins=bins)
        f_hs = histogram_specification(original_hist, reference_points[i], bins=bins)
        
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
    # Referencia por cada canal (R, G, B)
    reference_points_r = [
        (cv2.getTrackbarPos('R_P1_X', 'Ajuste de Histograma'), 0.0),
        (cv2.getTrackbarPos('R_P2_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('R_P2_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('R_P3_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('R_P3_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('R_P4_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('R_P4_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('R_P5_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('R_P5_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('R_P6_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('R_P6_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('R_P7_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('R_P7_Y', 'Ajuste de Histograma') / 100),
        (255, cv2.getTrackbarPos('R_P8_Y', 'Ajuste de Histograma') / 100)
    ]

    reference_points_g = [
        (cv2.getTrackbarPos('G_P1_X', 'Ajuste de Histograma'), 0.0),
        (cv2.getTrackbarPos('G_P2_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('G_P2_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('G_P3_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('G_P3_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('G_P4_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('G_P4_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('G_P5_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('G_P5_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('G_P6_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('G_P6_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('G_P7_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('G_P7_Y', 'Ajuste de Histograma') / 100),
        (255, cv2.getTrackbarPos('G_P8_Y', 'Ajuste de Histograma') / 100)
    ]

    reference_points_b = [
        (cv2.getTrackbarPos('B_P1_X', 'Ajuste de Histograma'), 0.0),
        (cv2.getTrackbarPos('B_P2_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('B_P2_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('B_P3_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('B_P3_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('B_P4_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('B_P4_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('B_P5_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('B_P5_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('B_P6_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('B_P6_Y', 'Ajuste de Histograma') / 100),
        (cv2.getTrackbarPos('B_P7_X', 'Ajuste de Histograma'), cv2.getTrackbarPos('B_P7_Y', 'Ajuste de Histograma') / 100),
        (255, cv2.getTrackbarPos('B_P8_Y', 'Ajuste de Histograma') / 100)
    ]

    # Ajustar imagen con las referencias de cada canal (R, G, B)
    reference_points = [reference_points_b, reference_points_g, reference_points_r]
    adjusted_image = apply_histogram_specification(img, reference_points)

    # Mostrar la imagen original y ajustada lado a lado
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
cv2.namedWindow('Ajuste de Histograma', cv2.WINDOW_NORMAL)

# Crear sliders para ajustar los puntos en X e Y de cada canal (R, G, B)
for channel in ['R', 'G', 'B']:
    cv2.createTrackbar(f'{channel}_P1_X', 'Ajuste de Histograma', 0, 255, update_histogram)
    cv2.createTrackbar(f'{channel}_P2_X', 'Ajuste de Histograma', 32, 255, update_histogram)
    cv2.createTrackbar(f'{channel}_P2_Y', 'Ajuste de Histograma', 25, 100, update_histogram)
    cv2.createTrackbar(f'{channel}_P3_X', 'Ajuste de Histograma', 64, 255, update_histogram)
    cv2.createTrackbar(f'{channel}_P3_Y', 'Ajuste de Histograma', 50, 100, update_histogram)
    cv2.createTrackbar(f'{channel}_P4_X', 'Ajuste de Histograma', 96, 255, update_histogram)
    cv2.createTrackbar(f'{channel}_P4_Y', 'Ajuste de Histograma', 75, 100, update_histogram)
    cv2.createTrackbar(f'{channel}_P5_X', 'Ajuste de Histograma', 128, 255, update_histogram)
    cv2.createTrackbar(f'{channel}_P5_Y', 'Ajuste de Histograma', 100, 100, update_histogram)
    cv2.createTrackbar(f'{channel}_P6_X', 'Ajuste de Histograma', 160, 255, update_histogram)
    cv2.createTrackbar(f'{channel}_P6_Y', 'Ajuste de Histograma', 75, 100, update_histogram)
    cv2.createTrackbar(f'{channel}_P7_X', 'Ajuste de Histograma', 192, 255, update_histogram)
    cv2.createTrackbar(f'{channel}_P7_Y', 'Ajuste de Histograma', 50, 100, update_histogram)
    cv2.createTrackbar(f'{channel}_P8_Y', 'Ajuste de Histograma', 25, 100, update_histogram)

# Llamar una vez para inicializar la visualización
update_histogram(0)

# Mantener la ventana abierta hasta que se presione la tecla 'q'
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

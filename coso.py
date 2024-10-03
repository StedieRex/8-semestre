import cv2
import numpy as np

def highlight_white_and_yellow(image):
    # Convertir a espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir rango para el color blanco en HSV
    lower_white = np.array([0, 0, 200])  # S bajo, V alto
    upper_white = np.array([180, 30, 255])  # No limitar el H, S bajo, V alto

    # Definir rango para el color amarillo en HSV
    lower_yellow = np.array([20, 100, 100])  # H ~50, S alto, V medio-alto
    upper_yellow = np.array([40, 255, 255])

    # Crear máscaras para blanco y amarillo
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Combinar las máscaras
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(image, image, mask=combined_mask)

    return result

# Cargar la imagen o frame
frame = cv2.imread('carretera-oscuro.png')

# Resaltar blanco y amarillo
highlighted_frame = highlight_white_and_yellow(frame)

# Mostrar la imagen resaltada
cv2.imshow('Blanco y Amarillo Resaltado', highlighted_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

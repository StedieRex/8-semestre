import cv2
import numpy as np
from tkinter import Tk
from tkinter import filedialog

# Función para abrir el cuadro de diálogo y seleccionar la imagen
def seleccionar_imagen():
    root = Tk()
    root.withdraw()
    ruta_imagen = filedialog.askopenfilename(title="Selecciona una imagen",
                                             filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp")])
    return ruta_imagen

# Función que se ejecuta cada vez que se mueve un slider
def actualizar_imagen(_):
    # Obtener los valores de los sliders (mínimos y máximos)
    hue_min = cv2.getTrackbarPos('Hue Min', 'Ajustes HSV')
    hue_max = cv2.getTrackbarPos('Hue Max', 'Ajustes HSV')
    saturation_min = cv2.getTrackbarPos('Saturation Min', 'Ajustes HSV')
    saturation_max = cv2.getTrackbarPos('Saturation Max', 'Ajustes HSV')
    value_min = cv2.getTrackbarPos('Value Min', 'Ajustes HSV')
    value_max = cv2.getTrackbarPos('Value Max', 'Ajustes HSV')

    # Crear una máscara basada en el rango HSV
    lower_bound = np.array([hue_min, saturation_min, value_min])
    upper_bound = np.array([hue_max, saturation_max, value_max])
    mascara = cv2.inRange(imagen_hsv, lower_bound, upper_bound)

    # Aplicar la máscara a la imagen original
    imagen_resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)

    # Mostrar la imagen modificada
    cv2.imshow('Imagen Modificada', imagen_resultado)

# Seleccionar la imagen con una ventana
ruta_imagen = seleccionar_imagen()

if ruta_imagen:
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)

    # Convertir la imagen de BGR a HSV
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Crear una ventana para los sliders
    cv2.namedWindow('Ajustes HSV')

    # Crear sliders para ajustar los valores mínimos y máximos de Hue, Saturation y Value
    cv2.createTrackbar('Hue Min', 'Ajustes HSV', 0, 360, actualizar_imagen)  # Hue Mínimo entre 0 y 360
    cv2.createTrackbar('Hue Max', 'Ajustes HSV', 360, 360, actualizar_imagen)  # Hue Máximo entre 0 y 360
    cv2.createTrackbar('Saturation Min', 'Ajustes HSV', 0, 255, actualizar_imagen)  # Saturation Mínimo entre 0 y 255
    cv2.createTrackbar('Saturation Max', 'Ajustes HSV', 255, 255, actualizar_imagen)  # Saturation Máximo entre 0 y 255
    cv2.createTrackbar('Value Min', 'Ajustes HSV', 0, 255, actualizar_imagen)  # Value Mínimo entre 0 y 255
    cv2.createTrackbar('Value Max', 'Ajustes HSV', 255, 255, actualizar_imagen)  # Value Máximo entre 0 y 255

    # Mostrar la imagen original para referencia
    cv2.imshow('Imagen Original', imagen)

    # Llamar la función de actualización de la imagen por primera vez
    actualizar_imagen(0)

    # Esperar indefinidamente hasta que el usuario presione una tecla
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No se seleccionó ninguna imagen.")

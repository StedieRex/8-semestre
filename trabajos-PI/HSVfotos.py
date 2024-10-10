import cv2
import numpy as np
import pickle
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

    # Obtener los valores de los sliders de contraste, gamma y brillo
    contraste = cv2.getTrackbarPos('Contraste', 'Ajustes HSV') / 50.0  # Rango de contraste ajustado a 0.0 - 2.0
    gamma = cv2.getTrackbarPos('Gamma', 'Ajustes HSV') / 100.0 + 0.1  # Rango de gamma ajustado a 0.1 - 1.1
    brillo = cv2.getTrackbarPos('Brillo', 'Ajustes HSV') - 100  # Rango de brillo ajustado a -100 a 100

    # Crear una máscara basada en el rango HSV
    lower_bound = np.array([hue_min, saturation_min, value_min])
    upper_bound = np.array([hue_max, saturation_max, value_max])
    mascara = cv2.inRange(imagen_hsv, lower_bound, upper_bound)

    # Aplicar la máscara a la imagen original
    imagen_resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)

    # Ajustar el contraste
    imagen_resultado = cv2.convertScaleAbs(imagen_resultado, alpha=contraste, beta=brillo)

    # Aplicar corrección gamma
    imagen_resultado = ajustar_gamma(imagen_resultado, gamma)

    # Obtener el tamaño de la ventana desde los sliders
    ancho = cv2.getTrackbarPos('Ancho', 'Ajustes HSV')
    alto = cv2.getTrackbarPos('Alto', 'Ajustes HSV')

    # Cambiar el tamaño de la imagen de resultado
    imagen_resultado = cv2.resize(imagen_resultado, (ancho, alto))

    # Mostrar la imagen modificada
    cv2.imshow('Imagen Modificada', imagen_resultado)

# Función para ajustar la corrección gamma
def ajustar_gamma(imagen, gamma=1.0):
    # Crear una tabla de corrección gamma
    tabla = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(imagen, tabla)

# Función para guardar la configuración de los sliders
def guardar_configuracion():
    config = {
        'hue_min': cv2.getTrackbarPos('Hue Min', 'Ajustes HSV'),
        'hue_max': cv2.getTrackbarPos('Hue Max', 'Ajustes HSV'),
        'saturation_min': cv2.getTrackbarPos('Saturation Min', 'Ajustes HSV'),
        'saturation_max': cv2.getTrackbarPos('Saturation Max', 'Ajustes HSV'),
        'value_min': cv2.getTrackbarPos('Value Min', 'Ajustes HSV'),
        'value_max': cv2.getTrackbarPos('Value Max', 'Ajustes HSV'),
        'contraste': cv2.getTrackbarPos('Contraste', 'Ajustes HSV'),
        'gamma': cv2.getTrackbarPos('Gamma', 'Ajustes HSV'),
        'brillo': cv2.getTrackbarPos('Brillo', 'Ajustes HSV'),
        'ancho': cv2.getTrackbarPos('Ancho', 'Ajustes HSV'),
        'alto': cv2.getTrackbarPos('Alto', 'Ajustes HSV')
    }
    
    archivo_guardar = filedialog.asksaveasfilename(defaultextension=".pkl", 
                                                   filetypes=[("Archivo de configuración", "*.pkl")])
    
    if archivo_guardar:
        with open(archivo_guardar, 'wb') as f:
            pickle.dump(config, f)
        print(f"Configuración guardada en {archivo_guardar}")

# Función para cargar la configuración de los sliders
def cargar_configuracion():
    archivo_cargar = filedialog.askopenfilename(title="Selecciona un archivo de configuración",
                                                filetypes=[("Archivo de configuración", "*.pkl")])
    
    if archivo_cargar:
        with open(archivo_cargar, 'rb') as f:
            config = pickle.load(f)
        
        # Aplicar la configuración a los sliders
        cv2.setTrackbarPos('Hue Min', 'Ajustes HSV', config['hue_min'])
        cv2.setTrackbarPos('Hue Max', 'Ajustes HSV', config['hue_max'])
        cv2.setTrackbarPos('Saturation Min', 'Ajustes HSV', config['saturation_min'])
        cv2.setTrackbarPos('Saturation Max', 'Ajustes HSV', config['saturation_max'])
        cv2.setTrackbarPos('Value Min', 'Ajustes HSV', config['value_min'])
        cv2.setTrackbarPos('Value Max', 'Ajustes HSV', config['value_max'])
        cv2.setTrackbarPos('Contraste', 'Ajustes HSV', config['contraste'])
        cv2.setTrackbarPos('Gamma', 'Ajustes HSV', config['gamma'])
        cv2.setTrackbarPos('Brillo', 'Ajustes HSV', config['brillo'])
        cv2.setTrackbarPos('Ancho', 'Ajustes HSV', config['ancho'])
        cv2.setTrackbarPos('Alto', 'Ajustes HSV', config['alto'])

        print(f"Configuración cargada desde {archivo_cargar}")

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

    # Crear sliders para ajustar contraste, gamma y brillo
    cv2.createTrackbar('Contraste', 'Ajustes HSV', 50, 100, actualizar_imagen)  # Contraste entre 0.0 y 2.0 (multiplicador)
    cv2.createTrackbar('Gamma', 'Ajustes HSV', 10, 100, actualizar_imagen)  # Gamma entre 0.1 y 1.1
    cv2.createTrackbar('Brillo', 'Ajustes HSV', 100, 200, actualizar_imagen)  # Brillo entre -100 y 100

    # Crear sliders para ajustar el tamaño de la ventana
    cv2.createTrackbar('Ancho', 'Ajustes HSV', 640, 1920, actualizar_imagen)  # Ancho máximo 1920
    cv2.createTrackbar('Alto', 'Ajustes HSV', 480, 1080, actualizar_imagen)  # Alto máximo 1080

    # Mostrar la imagen original
    cv2.imshow('Imagen Modificada', imagen)

    while True:
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('g'):
            guardar_configuracion()
        elif tecla == ord('l'):
            cargar_configuracion()
        elif tecla == 27:  # ESC
            break

cv2.destroyAllWindows()
 
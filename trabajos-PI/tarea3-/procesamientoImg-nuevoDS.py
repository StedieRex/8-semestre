import cv2
import numpy as np
import os
from tkinter import Tk, filedialog

# Función para crear una carpeta si no existe
def asegurar_directorio(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)

# Configurar y abrir diálogo para seleccionar la carpeta raíz
root = Tk()
root.withdraw()  # Ocultar ventana principal
carpeta_origen = filedialog.askdirectory(title="Seleccionar la carpeta raíz del dataset")
if not carpeta_origen:
    print("No se seleccionó ninguna carpeta.")
    exit()

# Configurar carpeta de salida
carpeta_destino = carpeta_origen + "_procesado"
asegurar_directorio(carpeta_destino)

# Subcarpetas y clases
subcarpetas = ['train', 'validation', 'test']
clases = ['bcc', 'nv']  # Puedes agregar más clases en esta lista en el futuro

# Procesar imágenes
for subcarpeta in subcarpetas:
    for clase in clases:
        # Definir rutas para la carpeta de origen y destino
        carpeta_clase_origen = os.path.join(carpeta_origen, subcarpeta, clase)
        carpeta_clase_destino = os.path.join(carpeta_destino, subcarpeta, clase)
        asegurar_directorio(carpeta_clase_destino)

        # Verificar si la carpeta de la clase existe antes de procesar
        if not os.path.exists(carpeta_clase_origen):
            print(f"La carpeta {carpeta_clase_origen} no existe, se omitirá.")
            continue

        # Recorrer archivos de la carpeta de la clase
        for archivo in os.listdir(carpeta_clase_origen):
            ruta_imagen_origen = os.path.join(carpeta_clase_origen, archivo)
            ruta_imagen_destino = os.path.join(carpeta_clase_destino, archivo)

            if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Leer imagen y convertir a escala de grises
                img = cv2.imread(ruta_imagen_origen, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (150, 150))

                # Transformada de Fourier
                dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)

                # Magnitud
                magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

                # Crear filtro pasa-bajas
                rows, cols = img.shape
                crow, ccol = rows // 2, cols // 2
                mask = np.zeros((rows, cols, 2), np.uint8)
                r = 20  # Radio del círculo
                center = [crow, ccol]
                x, y = np.ogrid[:rows, :cols]
                mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
                mask[mask_area] = 1

                # Aplicar máscara y transformada inversa
                fshift = dft_shift * mask
                f_ishift = np.fft.ifftshift(fshift)
                img_back = cv2.idft(f_ishift)
                img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

                # Aplicar filtro gaussiano
                img_back = cv2.GaussianBlur(img_back, (11, 11), 0)

                # Guardar imagen procesada
                img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)# esto es para que la imagen se vea bien
                cv2.imwrite(ruta_imagen_destino, img_back)
                

print(f"Procesamiento completado. Imágenes guardadas en: {carpeta_destino}")

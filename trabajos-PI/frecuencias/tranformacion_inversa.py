import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import tkinter as tk 
from tkinter import filedialog 
from PIL import Image, ImageTk 

# Crear ventana oculta para usar el diálogo 
root = tk.Tk() 
root.withdraw() # Oculta la ventana principal 

# Abrir el diálogo para seleccionar múltiples imágenes 
archivos_seleccionados = filedialog.askopenfilenames(title="Seleccionar imágenes", filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]) 
if archivos_seleccionados: 
    for archivo in archivos_seleccionados: 
        print(f"Archivo seleccionado: {archivo}") 
        ruta_imagen = archivo 
else: 
    print("No se seleccionó ningún archivo.") 
    
# Abrir la imagen y convertirla a escala de grises 
img = cv2.imread( ruta_imagen, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (400, 400))

#aplicar la transformada de fourier
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

#calcular la magnitud
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# crear un filtro  pasa-bajas (bloquear altas frecuencias)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
r = 30 # radio del circulo
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

# aplicar la mascara y la transformada inversa
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Visualizar la imagen original y la imagen filtrada
plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('imagen original'), plt.axis('off')
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('espectro de magnitud'), plt.axis('off')
plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('imagen filtrada'), plt.axis('off')
plt.show()
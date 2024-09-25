import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-k", "--mask", required=True, help="Size of the kernel")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
Original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# serpara los canales de la imagen
B, G, R = cv2.split(image)

#crear el kernel de k x k
k = int(args["mask"])
kernel = np.ones((k, k), np.float32)/(k*k) 

# Aplicar el filtro de suavizado
G = cv2.filter2D(G, -1, kernel)
R = cv2.filter2D(R, -1, kernel)

# comparacion de las imagenes
fig = plt.figure(figsize=(14, 5))# figsize=(14, 5) tamaño de la figura
ax1 = fig.add_subplot(1,2,1)# add_subplot(1,2,1) 1 fila, 2 columnas, 1 posicion
ax2 = fig.add_subplot(1,2,2)

ax1.imshow(Original)
ax1.set_title("Original")

image = cv2.merge([R, G, B])# se unen los canales de la imagen
ax2.imshow(image)
ax2.set_title("Suavizado")

plt.show()
#cv2.imshow("Original", Original)
#cv2.imshow("Suavizado", image)
#cv2.waitKey(0)


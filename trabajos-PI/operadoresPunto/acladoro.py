import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Main progrm
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the imput image")
args = vars(parser.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

# Se aplica el operador de aclarador a la imagen
# optener el valor maximo de la imagen de cada valor
#logarimo natural en python es np.log()
r = image_RGB.max()
c = 255.0/np.log(1 + r)
image_RGB_Aclarador =np.array( c*np.log(1 + image_RGB), dtype='uint8')

# Se genera una figura para mostrar los resultados con matploitlib
fig = plt.figure(figsize=(14, 10))
# Se maqueta el diseño del grafico
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,3,5)
# Se dibuja la imagen original
ax1.imshow(image_RGB)
ax1.set_title("Original image")
# Se dibuja la imagen con el operador
ax2.imshow(image_RGB_Aclarador)
ax2.set_title("Negative image")
# Se dibujan las graficas de las funciones
x = np.linspace(0, 255, 255)#se crea un array de 255 elementos que va desde 0 a 255
y1 = x
c2 = 255.0/np.log(1 + r)
y2 = np.array( c2*np.log(1 + x))
ax3.plot(x, y1, color="r", linewidth=1, label="f(x) = x (ID)")
msg = "Negative func."
ax3.plot(x, y2, color="b", linewidth=1, label=msg)
ax3.legend()
ax3.set_title("Negative function")
plt.show()
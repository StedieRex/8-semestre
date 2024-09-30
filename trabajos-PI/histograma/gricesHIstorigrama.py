import argparse
import matplotlib.pyplot as plt
import cv2

# Main program
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(parser.parse_args())

image_BGR = cv2.imread(args["image"])
image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)
image_BGR = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

# Se aplica el operador de aclarador a la imagen
hist = cv2.calcHist([image],[0],None,[256],[0,256])# none, es para aplicar la mascara o region especifica donde se quiere aplicar el histograma

# Se genera una figura para mostrar los resultados con matplotlib
fig = plt.figure(figsize=(14, 5))
# Se maqueta el diseño del grafico
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(1,2,2)
# Se dibuja la imagen original
ax1.imshow(image, cmap='gray')
ax1.set_title("Original image")
# Se dibuja la imagen con el operador
ax2.plot(hist)
ax2.set_title("Histogram")

plt.show()
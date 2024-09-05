import argparse
import cv2
import matplotlib.pyplot as plt

# Main progrm
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the imput image")
args = vars(parser.parse_args())

image_BGR = cv2.imread(args["image"])
image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
b,g,r = cv2.split(image)

# Se aplica el operador de aclarador a la imagen
hist = cv2.calcHist([r],[0],None,[255],[0,256])
hist = cv2.calcHist([g],[0],None,[255],[0,256])
hist = cv2.calcHist([b],[0],None,[255],[0,256])

# Se genera una figura para mostrar los resultados con matploitlib
fig = plt.figure(figsize=(14, 5))
# Se maqueta el diseño del grafico
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(1,2,2)
# Se dibuja la imagen original
ax1.imshow(image, cmap='gray')
ax1.set_title("Original image")
# Se dibuja la imagen con el operador
ax2.imshow(hist)
ax2.set_title("histogram")
import numpy as np
import argparse
import cv2

h_min_blanco = 0
h_max_blanco = 180  # Hue en OpenCV va de 0 a 180
s_min_blanco = 0
s_max_blanco = 60  # Saturación baja para captar el blanco
v_min_blanco = 200
v_max_blanco = 255  # Valores altos de brillo

# Rango para amarillo
h_min_amarillo = 0
h_max_amarillo = 100  # Tono específico del amarillo
s_min_amarillo = 100
s_max_amarillo = 255  # Saturación más alta para amarillo
v_min_amarillo = 120
v_max_amarillo = 255  

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",required=True,help="path to the image")
args=vars(ap.parse_args())

size = 11 #tamaño del kernel

image = cv2.imread(args["image"])
cv2.imshow("Original",image)

acentuando = image.copy()

hsv = cv2.cvtColor(acentuando, cv2.COLOR_BGR2HSV)

mascara_blanco = cv2.inRange(hsv, (h_min_blanco, s_min_blanco, v_min_blanco), (h_max_blanco, s_max_blanco, v_max_blanco))
mascara_amarillo = cv2.inRange(hsv, (h_min_amarillo, s_min_amarillo, v_min_amarillo), (h_max_amarillo, s_max_amarillo, v_max_amarillo))
mascara = cv2.bitwise_or(mascara_blanco, mascara_amarillo)
gray = cv2.cvtColor(acentuando,cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray,(size,size),0)
canny = cv2.Canny(blurred, 30, 120)

#cv2.imshow("Edge detection", np.hstack([canny,blurred,gray]))
cv2.imshow("Edge detection", np.hstack([gray,blurred,canny]))
cv2.waitKey(0)
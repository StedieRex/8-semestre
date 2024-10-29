import argparse
import cv2
import matplotlib.pyplot as plt

# Cargar la imagen
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-t", "--type", required=True, help="Type of filter")
ap.add_argument("-k", "--mask", required=True, help="Size of the kernel")
args = vars(ap.parse_args())

image = plt.imread(args["image"])
blurtype = int(args["type"])
size = int(args["mask"])

plt.figure(num="original")
plt.imshow(image,cmap='gray')

if blurtype == 0:
    plt.figure(num="avarage blurring")
    plt.imshow(cv2.blur(image,(size,size)))
elif blurtype == 1:
    plt.figure(num="gausian blurring")
    plt.imshow(cv2.GaussianBlur(image,(size,size),0))
elif blurtype == 2:
    plt.figure(num="median Blurring")
    plt.imshow(cv2.medianBlur(image,size))
elif blurtype == 3:
    plt.figure(num="Bilateral blurring")
    plt.imshow(cv2.bilateralFilter(image,size,21,21))
    
plt.show()
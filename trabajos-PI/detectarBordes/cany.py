import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",required=True,help="path to the image")
args=vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original",image)

gray = image.copy()
gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray,(11,11),0)
canny = cv2.Canny(blurred, 30, 120)

cv2.imshow("Edge detection", np.hstack([gray,blurred,canny]))
#cv2.imshow("Edge detection", np.hstack([canny,blurred,gray]))
cv2.waitKey(0)
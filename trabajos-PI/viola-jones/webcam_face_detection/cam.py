# USAGE
# python cam.py --face cascades/haarcascade_frontalface_default.xml
# python cam.py --face cascades/haarcascade_frontalface_default.xml --video video/adrian_face.mov

# import the necessary packages
from pyimagesearch.facedetector import FaceDetector
from pyimagesearch import imutils
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True,
	help = "path to where the face cascade resides")
ap.add_argument("-v", "--video",
	help = "path to the (optional) video file")
args = vars(ap.parse_args())

# construct the face detector
fd = FaceDetector(args["face"])

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # Convertir el frame a escala de grises
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    frameClone = frame.copy()

    # loop over the face bounding boxes and apply bilateral filter
    for (fX, fY, fW, fH) in faceRects:
        # Extraer la región del rostro de la imagen en escala de grises
        faceROI = gray[fY:fY + fH, fX:fX + fW]

        # Aplicar suavizado bilateral en el rostro
        enhancedFace = cv2.bilateralFilter(faceROI, d=5, sigmaColor=21, sigmaSpace=21)

        # Colocar el rostro suavizado en la copia del frame en color
        frameClone[fY:fY + fH, fX:fX + fW] = cv2.cvtColor(enhancedFace, cv2.COLOR_GRAY2BGR)

        # Dibujar el rectángulo del rostro en la imagen suavizada
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

    # Mostrar el video procesado en su resolución original
    cv2.imshow("Enhanced Face", frameClone)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

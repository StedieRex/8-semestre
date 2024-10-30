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

    # Aumentar el brillo del fotograma
    bright_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)  # Ajusta `alpha` para el contraste y `beta` para el brillo

    # Aplicar suavizado gaussiano
    smoothed_frame = cv2.GaussianBlur(bright_frame, (5,5), 0)

    # Convertir el fotograma suavizado a escala de grises
    smoothed_frame = imutils.resize(smoothed_frame, width=700)
    gray = cv2.cvtColor(smoothed_frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen suavizada y con brillo
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    frameClone = smoothed_frame.copy()

    # loop over the face bounding boxes (solo dibujo los rectángulos en este caso)
    for (fX, fY, fW, fH) in faceRects:
        # Dibujar el rectángulo del rostro en la imagen procesada
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

    # Mostrar el video procesado con brillo y suavizado gaussiano aplicados fuera del ciclo
    cv2.imshow("Enhanced Face", frameClone)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

import cv2
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def camara():
    # Rango para resaltar el color blanco en HSV
    h_min = 0
    h_max = 180  # Hue en OpenCV va de 0 a 180
    s_min = 0
    s_max = 60  # Saturación baja para captar el blanco
    v_min = 200
    v_max = 255  # Valores altos de brillo

    def cargar_video():
        Tk().withdraw()  # Ocultar la ventana principal de tkinter
        file_path = askopenfilename(filetypes=[("Video files", "*.mp4")])
        return file_path

    # Cargar el archivo de video
    video_path = cargar_video()
    if not video_path:
        print("No se seleccionó ningún archivo de video.")
        return

    # Captura de video
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error al abrir el video")
        exit()

    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

            # Aplicar los valores HSV para resaltar el color blanco
            mascara = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
            result = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mascara)

            # Mostrar la imagen enmascarada y el video original
            cv2.imshow('Original', frame)
            cv2.imshow('Resaltado Blanco', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camara()
    print("Fin del programa")

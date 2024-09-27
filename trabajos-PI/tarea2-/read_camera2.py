import cv2
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def camara():
    # Rango para resaltar el color blanco y amarillo en HSV
    # Blanco: Baja saturación, alto brillo
    # Amarillo: Saturación moderada, tono amarillo
    h_min_blanco = 0
    h_max_blanco = 180  # Hue en OpenCV va de 0 a 180
    s_min_blanco = 0
    s_max_blanco = 60  # Saturación baja para captar el blanco
    v_min_blanco = 200
    v_max_blanco = 255  # Valores altos de brillo

    # Rango para amarillo
    h_min_amarillo = 10
    h_max_amarillo = 30  # Tono específico del amarillo
    s_min_amarillo = 100
    s_max_amarillo = 255  # Saturación más alta para amarillo
    v_min_amarillo = 150
    v_max_amarillo = 255  # Amarillo es brillante también

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
            mascara_blanco = cv2.inRange(hsv, (h_min_blanco, s_min_blanco, v_min_blanco), (h_max_blanco, s_max_blanco, v_max_blanco))

            # Aplicar los valores HSV para resaltar el color amarillo
            mascara_amarillo = cv2.inRange(hsv, (h_min_amarillo, s_min_amarillo, v_min_amarillo), (h_max_amarillo, s_max_amarillo, v_max_amarillo))

            # Combinar ambas máscaras (blanco y amarillo)
            mascara = cv2.bitwise_or(mascara_blanco, mascara_amarillo)
            result = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mascara)

            # Mostrar la imagen enmascarada y el video original
            cv2.imshow('Original', frame)
            cv2.imshow('Resaltado Blanco y Amarillo', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camara()
    print("Fin del programa")

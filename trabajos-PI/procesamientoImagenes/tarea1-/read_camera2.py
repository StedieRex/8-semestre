import cv2
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def camara():
    # Valores iniciales para HSV
    h_min = 0
    h_max = 179
    s_min = 0
    s_max = 255
    v_min = 0
    v_max = 255
    
    def update(val):
        nonlocal h_min, h_max, s_min, s_max, v_min, v_max
        h_min = int(hmin.val)
        h_max = int(hmax.val)
        s_min = int(smin.val)
        s_max = int(smax.val)
        v_min = int(vmin.val)
        v_max = int(vmax.val)
    
    # Configuración del ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("index_camera", help="Índice de la cámara para leer", type=int)
    args = parser.parse_args()

    # Captura de video
    capture = cv2.VideoCapture(args.index_camera)
    if not capture.isOpened():
        print("Error al abrir la cámara")
        exit()
    
    # Configuración de los sliders
    plt.ion()
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.4)
    axhmin = plt.axes([0.25, 0.3, 0.65, 0.03])
    axhmax = plt.axes([0.25, 0.25, 0.65, 0.03])
    axsmin = plt.axes([0.25, 0.2, 0.65, 0.03])
    axsmax = plt.axes([0.25, 0.15, 0.65, 0.03])
    axvmin = plt.axes([0.25, 0.1, 0.65, 0.03])
    axvmax = plt.axes([0.25, 0.05, 0.65, 0.03])
    
    hmin = Slider(axhmin, 'H min', 0, 179, valinit=h_min)
    hmax = Slider(axhmax, 'H max', 0, 179, valinit=h_max)
    smin = Slider(axsmin, 'S min', 0, 255, valinit=s_min)
    smax = Slider(axsmax, 'S max', 0, 255, valinit=s_max)
    vmin = Slider(axvmin, 'V min', 0, 255, valinit=v_min)
    vmax = Slider(axvmax, 'V max', 0, 255, valinit=v_max)

    hmin.on_changed(update)
    hmax.on_changed(update)
    smin.on_changed(update)
    smax.on_changed(update)
    vmin.on_changed(update)
    vmax.on_changed(update)

    ret, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = ax.imshow(frame)

    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
            result = cv2.bitwise_and(frame, frame, mask=mask)

            im.set_data(result)
            plt.draw()
            plt.pause(0.01)
            
            # Salir del programa
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    print("Saliendo del programa")
    capture.release()
    cv2.destroyAllWindows()
    plt.show()

if __name__ == "__main__":
    camara()
    print("Fin del programa")
    plt.close()

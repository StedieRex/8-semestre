import cv2
import numpy as np
from tkinter import Tk, Toplevel, Scale, HORIZONTAL, VERTICAL, Frame, Scrollbar, Canvas
from tkinter.filedialog import askopenfilename, asksaveasfilename

def compute_histogram(image_channel, bins=256):
    hist, _ = np.histogram(image_channel, bins=bins, range=(0, bins))
    return hist

def compute_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf.max())
    return cdf_normalized

def histogram_specification(original_hist, reference_points, bins=256):
    P_A = compute_cdf(original_hist)
    K = bins
    f_hs = np.zeros(K, dtype=np.float32)

    for a in range(K):
        b = P_A[a]
        if b <= reference_points[0][1]:
            a_prime = reference_points[0][0]
        elif b >= reference_points[-1][1]:
            a_prime = reference_points[-1][0]
        else:
            n = len(reference_points) - 1
            while n >= 0 and b < reference_points[n][1]:
                n -= 1
            n = max(n, 0)
            a_prime = reference_points[n][0] + (b - reference_points[n][1]) * \
                      ((reference_points[n + 1][0] - reference_points[n][0]) / 
                      (reference_points[n + 1][1] - reference_points[n][1]))
        f_hs[a] = np.clip(a_prime, 0, 255)

    return f_hs

def apply_histogram_specification(image, reference_points, bins=256):
    channels = cv2.split(image)
    result_channels = []

    for i, channel in enumerate(channels):
        original_hist = compute_histogram(channel, bins=bins)
        f_hs = histogram_specification(original_hist, reference_points[i], bins=bins)
        
        new_channel = np.clip(np.interp(channel.flatten(), np.arange(bins), f_hs), 0, 255).reshape(channel.shape)
        result_channels.append(new_channel.astype(np.uint8))
    
    result_image = cv2.merge(result_channels)
    return result_image

def select_image():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    root.destroy()
    return file_path

def update_histogram(val=None):
    reference_points_r = [
        (red_sliders[0].get(), 0.0),
        (red_sliders[1].get(), red_sliders[2].get() / 100),
        (red_sliders[3].get(), red_sliders[4].get() / 100),
        (red_sliders[5].get(), red_sliders[6].get() / 100),
        (red_sliders[7].get(), red_sliders[8].get() / 100),
        (red_sliders[9].get(), red_sliders[10].get() / 100),
        (red_sliders[11].get(), red_sliders[12].get() / 100),
        (255, red_sliders[13].get() / 100)
    ]

    reference_points_g = [
        (green_sliders[0].get(), 0.0),
        (green_sliders[1].get(), green_sliders[2].get() / 100),
        (green_sliders[3].get(), green_sliders[4].get() / 100),
        (green_sliders[5].get(), green_sliders[6].get() / 100),
        (green_sliders[7].get(), green_sliders[8].get() / 100),
        (green_sliders[9].get(), green_sliders[10].get() / 100),
        (green_sliders[11].get(), green_sliders[12].get() / 100),
        (255, green_sliders[13].get() / 100)
    ]

    reference_points_b = [
        (blue_sliders[0].get(), 0.0),
        (blue_sliders[1].get(), blue_sliders[2].get() / 100),
        (blue_sliders[3].get(), blue_sliders[4].get() / 100),
        (blue_sliders[5].get(), blue_sliders[6].get() / 100),
        (blue_sliders[7].get(), blue_sliders[8].get() / 100),
        (blue_sliders[9].get(), blue_sliders[10].get() / 100),
        (blue_sliders[11].get(), blue_sliders[12].get() / 100),
        (255, blue_sliders[13].get() / 100)
    ]

    reference_points = [reference_points_b, reference_points_g, reference_points_r]
    adjusted_image = apply_histogram_specification(img, reference_points)

    combined_image = np.hstack((img, adjusted_image))
    cv2.imshow('Ajuste de Histograma', combined_image)

# Cargar la imagen
image_path = select_image()
if not image_path:
    raise FileNotFoundError("No se seleccionó ninguna imagen")

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("La imagen no pudo ser cargada")

# Crear la ventana de Tkinter para los sliders
root = Tk()
root.title("Control de Sliders")

# Crear un frame con scrollbar para contener los sliders
container = Frame(root)
canvas = Canvas(container)
scrollbar = Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Empaquetar el frame con el scrollbar
container.pack(fill="both", expand=True)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Crear sliders para cada canal
def create_sliders(parent, label):
    sliders = []
    for i in range(1, 9):
        sliders.append(Scale(parent, from_=0, to=255, orient=HORIZONTAL, label=f'{label}_P{i}_X'))
        sliders[-1].pack()
        sliders.append(Scale(parent, from_=0, to=100, orient=HORIZONTAL, label=f'{label}_P{i}_Y'))
        sliders[-1].pack()
    return sliders

# Crear sliders para cada canal y empaquetarlos
red_sliders = create_sliders(scrollable_frame, "R")
green_sliders = create_sliders(scrollable_frame, "G")
blue_sliders = create_sliders(scrollable_frame, "B")

# Mostrar la ventana de OpenCV con la imagen
cv2.namedWindow('Ajuste de Histograma', cv2.WINDOW_NORMAL)
update_histogram()

# Conectar los sliders con la función de actualización
for slider in red_sliders + green_sliders + blue_sliders:
    slider.config(command=update_histogram)

# Ejecutar el loop de Tkinter
root.mainloop()

# Cerrar OpenCV cuando se cierre la ventana de Tkinter
cv2.destroyAllWindows()

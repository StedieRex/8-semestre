import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

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

    for channel in channels:
        original_hist = compute_histogram(channel, bins=bins)
        f_hs = histogram_specification(original_hist, reference_points, bins=bins)
        
        new_channel = np.clip(np.interp(channel.flatten(), np.arange(bins), f_hs), 0, 255).reshape(channel.shape)
        result_channels.append(new_channel.astype(np.uint8))
    
    masked_frame = cv2.merge(result_channels)
    return masked_frame

def select_video():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    root.destroy()
    return file_path

video_path = select_video()

if not video_path:
    raise FileNotFoundError("No se seleccionó ningún video")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError("El video no pudo ser cargado")

reference_points = [
    (0, 0.0),
    (15, 0.57),  
    (29, 0.64),  
    (36, 0.77),  
    (164, 0.78),  
    (177, 0.78),  
    (209, 0.91),  
    (255, 1.0)
]

h_min_blanco = 0
h_max_blanco = 180
s_min_blanco = 0
s_max_blanco = 60
v_min_blanco = 200
v_max_blanco = 255

h_min_amarillo = 10
h_max_amarillo = 30
s_min_amarillo = 100
s_max_amarillo = 255
v_min_amarillo = 150
v_max_amarillo = 255

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    masked_frame = apply_histogram_specification(frame, reference_points)

    # hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
    # mascara_blanco = cv2.inRange(hsv, (h_min_blanco, s_min_blanco, v_min_blanco), (h_max_blanco, s_max_blanco, v_max_blanco))
    # mascara_amarillo = cv2.inRange(hsv, (h_min_amarillo, s_min_amarillo, v_min_amarillo), (h_max_amarillo, s_max_amarillo, v_max_amarillo))
    # mascara = cv2.bitwise_or(mascara_blanco, mascara_amarillo)
    # masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mascara)

    cv2.imshow('Video Original y Procesado', masked_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

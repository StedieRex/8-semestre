import os
import tkinter as tk
from tkinter import filedialog
import shutil
import math

def seleccionar_carpeta():
    # Crear ventana oculta para usar el diálogo
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal

    # Abrir el diálogo para seleccionar carpeta
    carpeta_seleccionada = filedialog.askdirectory(title="Seleccionar carpeta")

    if carpeta_seleccionada:
        print(f"Carpeta seleccionada: {carpeta_seleccionada}")
        dividir_archivos(carpeta_seleccionada)
    else:
        print("No se seleccionó ninguna carpeta.")

def dividir_archivos(carpeta):
    try:
        # Listar archivos en la carpeta
        archivos = [f for f in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, f))]
        
        if not archivos:
            print("La carpeta seleccionada no contiene archivos.")
            return
        
        print(f"\nTotal de archivos encontrados: {len(archivos)}")

        # Calcular cantidades
        total_archivos = len(archivos)
        num_carpeta1 = math.floor(total_archivos * 0.7)
        num_carpeta2 = math.floor(total_archivos * 0.15)
        num_carpeta3 = total_archivos - num_carpeta1 - num_carpeta2  # El resto para la carpeta 3

        print(f"\nDistribución de archivos:")
        print(f"70% -> {num_carpeta1} archivos para carpeta 1")
        print(f"15% -> {num_carpeta2} archivos para carpeta 2")
        print(f"15% -> {num_carpeta3} archivos para carpeta 3")

        # Crear carpetas
        carpeta1 = os.path.join(carpeta, os.path.basename(carpeta) + "1")
        carpeta2 = os.path.join(carpeta, os.path.basename(carpeta) + "2")
        carpeta3 = os.path.join(carpeta, os.path.basename(carpeta) + "3")

        os.makedirs(carpeta1, exist_ok=True)
        os.makedirs(carpeta2, exist_ok=True)
        os.makedirs(carpeta3, exist_ok=True)

        # Mover archivos
        for i, archivo in enumerate(archivos):
            origen = os.path.join(carpeta, archivo)
            if i < num_carpeta1:
                destino = os.path.join(carpeta1, archivo)
            elif i < num_carpeta1 + num_carpeta2:
                destino = os.path.join(carpeta2, archivo)
            else:
                destino = os.path.join(carpeta3, archivo)
            shutil.move(origen, destino)

        print("\nArchivos distribuidos exitosamente.")
        print(f"Carpeta 1: {carpeta1}")
        print(f"Carpeta 2: {carpeta2}")
        print(f"Carpeta 3: {carpeta3}")

    except Exception as e:
        print(f"Error al dividir archivos: {e}")

if __name__ == "__main__":
    seleccionar_carpeta()

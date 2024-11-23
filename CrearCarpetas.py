import os
import tkinter as tk
from tkinter import filedialog

def seleccionar_carpeta():
    # Crear ventana oculta para usar el diálogo
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal

    # Abrir el diálogo para seleccionar carpeta
    carpeta_seleccionada = filedialog.askdirectory(title="Seleccionar carpeta")

    if carpeta_seleccionada:
        print(f"Carpeta seleccionada: {carpeta_seleccionada}")
        listar_archivos(carpeta_seleccionada)
    else:
        print("No se seleccionó ninguna carpeta.")

def listar_archivos(carpeta):
    print("\nArchivos en la carpeta seleccionada:")
    try:
        for archivo in os.listdir(carpeta):
            ruta_completa = os.path.join(carpeta, archivo)
            if os.path.isfile(ruta_completa):
                print(f"- {archivo}")
    except Exception as e:
        print(f"Error al listar archivos: {e}")

if __name__ == "__main__":
    seleccionar_carpeta()

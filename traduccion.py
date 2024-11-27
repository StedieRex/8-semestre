import tkinter as tk
from tkinter import filedialog
from deep_translator import GoogleTranslator

def traducir_srt():
    # Crear una ventana para seleccionar el archivo
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    ruta_archivo = filedialog.askopenfilename(
        title="Selecciona el archivo SRT",
        filetypes=[("Archivos SRT", "*.srt")]
    )
    
    if not ruta_archivo:
        print("No se seleccionó ningún archivo.")
        return

    # Crear el traductor
    traductor = GoogleTranslator(source='en', target='es')

    # Leer el archivo SRT
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        lineas = archivo.readlines()

    # Traducir las líneas manteniendo el formato
    lineas_traducidas = []
    for linea in lineas:
        if linea.strip().isdigit() or '-->' in linea or linea.strip() == "":
            # Mantener números, marcas de tiempo y líneas vacías sin traducir
            lineas_traducidas.append(linea)
        else:
            # Traducir el texto
            traduccion = traductor.translate(linea.strip())
            lineas_traducidas.append(traduccion + '\n')

    # Guardar el archivo traducido
    ruta_traducida = ruta_archivo.replace('.srt', '_traducido.srt')
    with open(ruta_traducida, 'w', encoding='utf-8') as archivo:
        archivo.writelines(lineas_traducidas)

    print(f"Archivo traducido guardado en: {ruta_traducida}")

# Ejecutar el programa
if __name__ == "__main__":
    traducir_srt()

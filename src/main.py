# main.py

import os
import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from procesador_imagen import ProcesadorImagen


def actualizar_imagen(root, label, procesador, imagenes_por_verdura, index):
    imagenes = imagenes_por_verdura[index]
    for col, img in enumerate(imagenes):
        img_resized = cv2.resize(img, (200, 200))
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label.config(image=img_tk)
        label.image = img_tk
        root.update()


if __name__ == "__main__":
    ruta_base = r'../anexos/imagenes_mias'
    carpetas = ['berenjena', 'camote', 'papa', 'zanahoria']

    root = tk.Tk()
    label = Label(root)
    label.pack()

    imagenes_por_verdura = []

    for carpeta in carpetas:
        ruta_carpeta = os.path.join(ruta_base, carpeta)
        if not os.path.exists(ruta_carpeta):
            print(f'La carpeta {ruta_carpeta} no existe.')
            continue

        archivos = os.listdir(ruta_carpeta)
        if not archivos:
            print(f'La carpeta {ruta_carpeta} está vacía.')
            continue

        for archivo in archivos:  # Iterar sobre todos los archivos
            ruta_imagen = os.path.join(ruta_carpeta, archivo)
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                print(f'No se pudo leer la imagen {ruta_imagen}.')
                continue

            procesador = ProcesadorImagen(imagen)
            filtros_a_aplicar = ['nln', 'gris', 'binarizada', 'morfologico', 'bordes']
            imagenes_filtradas = procesador.aplicar_filtros(filtros_a_aplicar)
            imagenes_por_verdura.append(imagenes_filtradas)

            # Mostrar la imagen y esperar la entrada del usuario
            actualizar_imagen(root, label, procesador, imagenes_por_verdura, len(imagenes_por_verdura) - 1)

            while True:
                print(f'Imagen: {ruta_imagen}')
                respuesta = input(
                    'Presione "S" para continuar con la siguiente imagen o "Q" para salir: ').strip().upper()
                if respuesta == "S":
                    break
                elif respuesta == "Q":
                    exit()


    root.mainloop()  # Ejecutar el bucle principal de Tkinter

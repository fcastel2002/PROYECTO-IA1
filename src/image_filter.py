import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from Archivos import *
from Parametros import calcular_momentos_hu, guardar_momentos_hu
import time  # Add this import
import threading  # Add this import
import sys  # Add this import
import select  # Add this import

class ProcesadorImagen:
    def __init__(self, imagen, etiqueta):
        self.imagen_original = imagen
        self.imagen = self.imagen_original.copy()
        self.etiqueta = etiqueta
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)


    def aplicar_filtro(self, filtro_nombre):
        if filtro_nombre == 'mediana':
            self.imagen = cv2.medianBlur(self.imagen, 7)
        elif filtro_nombre == 'gris':
            self.imagen = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        elif filtro_nombre == 'bordes':

            gris = self.imagen

            contornos, _ = cv2.findContours(gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contorno_mas_grande = max(contornos, key=cv2.contourArea)

            self.imagen = cv2.cvtColor(gris, cv2.COLOR_GRAY2RGB)

            self.imagen = cv2.drawContours(self.imagen, [contorno_mas_grande], -1, (0, 255, 0), 4)

            # Apply mask to remove background
            mask = np.zeros_like(gris)
            cv2.drawContours(mask, [contorno_mas_grande], -1, 255, thickness=cv2.FILLED)
            self.imagen = cv2.bitwise_and(self.imagen_original, self.imagen_original, mask=mask)

            # # Calcular y guardar momentos de Hu
            # hu_momentos = calcular_momentos_hu(contorno_mas_grande)
            # ruta_csv = 'momentos_hu.csv'
            # if not os.path.exists(ruta_csv):
            #     encabezados = ['Etiqueta'] + [f'Hu{i+1}' for i in range(7)]
            #     crear_archivo_csv(ruta_csv, encabezados)
            # guardar_momentos_hu(ruta_csv, self.etiqueta, hu_momentos)
           
        elif filtro_nombre == 'binarizada':

            gris = self.imagen
            self.imagen = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 3)

        elif filtro_nombre == 'nln':
            self.imagen = cv2.fastNlMeansDenoisingColored(self.imagen, None, 15, 5, 3, 14)

        elif filtro_nombre == 'morfologico':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            self.imagen = cv2.morphologyEx(self.imagen, cv2.MORPH_CLOSE, kernel)

        else:
            raise ValueError(f'Filtro "{filtro_nombre}" no reconocido')


    def aplicar_filtros(self, filtros):
        imagenes_filtradas = [self.imagen_original]
        for filtro in filtros:
            self.aplicar_filtro(filtro)
            imagenes_filtradas.append(self.imagen.copy())
        return imagenes_filtradas

def mostrar_imagenes(titulo, imagenes_por_verdura):
    ventana = tk.Toplevel()
    ventana.title(titulo)
    for row, imagenes in enumerate(imagenes_por_verdura):
        for col, img in enumerate(imagenes):
            img_resized = cv2.resize(img, (200, 200))
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            else:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            label = Label(ventana, image=img_tk)
            label.image = img_tk
            label.grid(row=row, column=col)
    return ventana  # Retorna la ventana para poder cerrarla luego

def guardar_imagenes_correctas(path, imagenes_por_verdura, carpetas,index):
    for i, carpeta in enumerate(carpetas):
        ruta_carpeta = os.path.join(path, carpeta)
        os.makedirs(ruta_carpeta, exist_ok=True)
        if imagenes_por_verdura[i]:  # Verificar que la lista no esté vacía
            ruta_imagen = os.path.join(ruta_carpeta, f'imagen_{index}.png')
            cv2.imwrite(ruta_imagen, imagenes_por_verdura[i][0])


if __name__ == "__main__":
    indice = 1
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    ventana_actual = None  # Variable para almacenar la ventana actual

    total_images = 12  # Total number of images in each folder

    while indice <= total_images:
        if ventana_actual is not None:
            ventana_actual.destroy()

        ruta_base = r'../anexos/imagenes_correctas'
        carpetas = ['berenjena', 'camote', 'papa', 'zanahoria']

        imagenes_originales = obtener_imagenes_por_verdura(ruta_base, carpetas, indice)
        if not any(imagen.size > 0 for imagen in imagenes_originales):  # Verificar si no hay más imágenes
            break
        indice += 1

        imagenes_por_verdura = []
        for imagen, etiqueta in zip(imagenes_originales, carpetas):
            procesador = ProcesadorImagen(imagen, etiqueta)
            filtros_a_aplicar = ['nln','mediana', 'gris', 'binarizada', 'morfologico', 'bordes']
            imagenes_filtradas = procesador.aplicar_filtros(filtros_a_aplicar)
            imagenes_por_verdura.append(imagenes_filtradas)

        # Mostrar todas las imágenes en una nueva ventana y guardar la referencia
        ventana_actual = mostrar_imagenes("Imágenes Filtradas", imagenes_por_verdura)
        root.update()

        input("Enter para continuar...")






import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from Archivos import *
from Parametros import *
import time  # Add this import
import threading  # Add this import
import sys  # Add this import
import select  # Add this import

class ProcesadorImagen:
    def __init__(self, imagen, etiqueta):
        self.imagen_original = imagen  # Ya está en BGR
        self.imagen = self.imagen_original.copy()
        self.etiqueta = etiqueta
        self.imagen_pre_filtrada = None
        self.imagen_sin_fondo = None


    def aplicar_filtro(self, filtro_nombre):
        if filtro_nombre == 'mediana':
            self.imagen = cv2.medianBlur(self.imagen, 11)
        elif filtro_nombre == 'gris':
            self.imagen = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        elif filtro_nombre == 'bordes':

            gris = self.imagen

            contornos, _ = cv2.findContours(gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contorno_mas_grande = max(contornos, key=cv2.contourArea)

            self.imagen = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

            self.imagen = cv2.drawContours(self.imagen, [contorno_mas_grande], -1, (0, 255, 0), 4)

            # Apply mask to remove background
            mask = np.zeros_like(gris)
            cv2.drawContours(mask, [contorno_mas_grande], -1, 255, thickness=cv2.FILLED)
            self.imagen = cv2.bitwise_and(self.imagen_original, self.imagen_original, mask=mask)
            
            # Apply 'saturacion' filter before calculating mean color
            self.aplicar_filtro('saturacion')
            
           
        elif filtro_nombre == 'binarizada':

            gris = self.imagen
            self.imagen = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        elif filtro_nombre == 'nln':
            self.imagen = cv2.fastNlMeansDenoisingColored(self.imagen, None, 15, 5, 3, 14)

        elif filtro_nombre == 'morfologico':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self.imagen = cv2.morphologyEx(self.imagen, cv2.MORPH_CLOSE, kernel)
            self.imagen_pre_filtrada = self.imagen.copy()

        elif filtro_nombre == 'saturacion':
            # Convert image to HSV color space
            hsv = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2HSV)
            # Increase saturation
            h, s, v = cv2.split(hsv)
            s = cv2.add(s, 70)  # Increase saturation by 50 (adjust as needed)
            s = np.clip(s, 0, 255)
    
            hsv_modified = cv2.merge([h, s, v])
            self.imagen = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)
            self.imagen_sin_fondo = self.imagen.copy()
        else:
            raise ValueError(f'Filtro "{filtro_nombre}" no reconocido')


    def aplicar_filtros(self, filtros):
        imagenes_filtradas = [self.imagen_original]
        for filtro in filtros:
            if filtro == 'bordes':
                # Guardar la imagen antes de aplicar el filtro de bordes
                self.imagen_pre_filtrada = self.imagen.copy()
            self.aplicar_filtro(filtro)
            imagenes_filtradas.append(self.imagen.copy())
        return imagenes_filtradas

def mostrar_imagenes(titulo, imagenes_por_verdura):
    ventana = tk.Toplevel()
    ventana.title(titulo)
    for row, imagenes in enumerate(imagenes_por_verdura):
        for col, img in enumerate(imagenes):
            img_resized = cv2.resize(img, (200, 200))
            # Convertir de BGR a RGB solo para mostrar
            if len(img_resized.shape) == 3:  # Si es una imagen en color
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

    total_images = 20  # Total number of images in each folder

    # Add this block to initialize the CSV file before processing begins
    ruta_csv = 'momentos_hu.csv'
    encabezados = ['Nombre', 'Hu2', 'Hu4', 'Hu6', 'Mean_B', 'Mean_G', 'Mean_R']
    crear_archivo_csv(ruta_csv, encabezados)
    
    carpetas = ['berenjena', 'camote', 'papa', 'zanahoria']
    ruta_base = r'../anexos/imagenes_correctas'
    ruta_imagenes_filtradas = r'../anexos/imagenes_filtradas'
    ruta_imagenes_sinfondo = r'../anexos/imagenes_sinfondo'
    os.makedirs(ruta_imagenes_filtradas, exist_ok=True)
    for carpeta in carpetas:
        ruta_carpeta_filtrada = os.path.join(ruta_imagenes_filtradas, carpeta)
        os.makedirs(ruta_carpeta_filtrada, exist_ok=True)

    while indice <= total_images:
        if ventana_actual is not None:
            ventana_actual.destroy()

        ruta_base = r'../anexos/imagenes_correctas'

        imagenes_originales = obtener_imagenes_por_verdura(ruta_base, carpetas, indice)
        if not any(imagen.size > 0 for imagen in imagenes_originales):  # Verificar si no hay más imágenes
            break
        indice += 1

        imagenes_por_verdura = []
        for imagen, etiqueta in zip(imagenes_originales, carpetas):
            procesador = ProcesadorImagen(imagen, etiqueta)
            filtros_a_aplicar = ['nln','mediana', 'gris', 'binarizada', 'bordes']
            imagenes_filtradas = procesador.aplicar_filtros(filtros_a_aplicar)
            
            # Crear carpetas si no existen
            ruta_carpeta = os.path.join(ruta_imagenes_filtradas, etiqueta)
            ruta_carpetaSF = os.path.join(ruta_imagenes_sinfondo, etiqueta)
            os.makedirs(ruta_carpetaSF, exist_ok=True)
            
            # Definir rutas de las imágenes
            ruta_imagen = os.path.join(ruta_carpeta, f'imagen_{indice}.png')
            ruta_imagenSF = os.path.join(ruta_carpetaSF, f'imagen_{indice}.png')
            
            # Verificar si las imágenes existen y no son None
            # if procesador.imagen_pre_filtrada is not None and procesador.imagen_sin_fondo is not None:
            #     if isinstance(procesador.imagen_pre_filtrada, np.ndarray) and isinstance(procesador.imagen_sin_fondo, np.ndarray):
            #         cv2.imwrite(ruta_imagen, procesador.imagen_pre_filtrada)
            #         cv2.imwrite(ruta_imagenSF, procesador.imagen_sin_fondo)
            
            imagenes_por_verdura.append(imagenes_filtradas)

        # Mostrar todas las imágenes en una nueva ventana y guardar la referencia
        ventana_actual = mostrar_imagenes("Imágenes Filtradas", imagenes_por_verdura)
        input("Enter para continuar...")
        root.update()

        #input("Enter para continuar...")






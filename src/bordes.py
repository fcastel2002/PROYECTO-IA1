import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from Archivos import *
from Parametros import *


class ProcesadorImagen:
    def __init__(self, imagen_filtrada, imagen_sinfondo, etiqueta):
        self.imagen_filtrada = imagen_filtrada
        self.imagen_sinfondo = imagen_sinfondo
        self.etiqueta = etiqueta
        self.imagen = imagen_filtrada.copy()
        self.momentos = [0,2,4,6]
        self.encabezados = ['Nombre'] + [f'Hu{i}' for i in self.momentos] + ['Mean_B', 'Mean_G', 'Mean_R']
        
    def aplicar_filtro(self, filtro_nombre):
        if filtro_nombre == 'bordes':
            gris = self.imagen

            contornos, _ = cv2.findContours(gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contorno_mas_grande = max(contornos, key=cv2.contourArea)

            self.imagen = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

            self.imagen = cv2.drawContours(self.imagen, [contorno_mas_grande], -1, (0, 255, 0), 4)

            # Apply mask to remove background
            mask = np.zeros_like(gris)
            cv2.drawContours(mask, [contorno_mas_grande], -1, 255, thickness=cv2.FILLED)
            self.imagen = cv2.bitwise_and(self.imagen_sinfondo, self.imagen_sinfondo, mask=mask)
            
            # Apply 'saturacion' filter before calculating mean color
            self.aplicar_filtro('saturacion')
            # Calculate mean color after increasing saturation using the contour mask
            mean_color = calcular_color_promedio(self.imagen, mask)

        # Calcular y guardar momentos de Hu
            hu_momentos = calcular_momentos_hu(contorno_mas_grande,self.momentos)
            ruta_csv = 'momentos_hu2.csv'
                
            
            guardar_momentos_hu(ruta_csv, self.etiqueta, hu_momentos, mean_color)
           
def obtener_imagenes_por_verdura(ruta_base_filtrada, ruta_base_sinfondo, carpetas):
    imagenes_filtradas = []
    imagenes_sinfondo = []
    
    for carpeta in carpetas:
        ruta_carpeta_filtrada = os.path.join(ruta_base_filtrada, carpeta)
        ruta_carpeta_sinfondo = os.path.join(ruta_base_sinfondo, carpeta)
        
        for archivo in os.listdir(ruta_carpeta_filtrada):
            ruta_imagen_filtrada = os.path.join(ruta_carpeta_filtrada, archivo)
            ruta_imagen_sinfondo = os.path.join(ruta_carpeta_sinfondo, archivo)
            
            print(f"Buscando imagen filtrada: {ruta_imagen_filtrada}")
            print(f"Buscando imagen sin fondo: {ruta_imagen_sinfondo}")
            
            if os.path.exists(ruta_imagen_filtrada) and os.path.exists(ruta_imagen_sinfondo):
                img_filtrada = cv2.imread(ruta_imagen_filtrada, cv2.IMREAD_GRAYSCALE)
                img_sinfondo = cv2.imread(ruta_imagen_sinfondo)
                
                if img_filtrada is None or img_sinfondo is None:
                    print(f"Error al cargar las imágenes para {archivo} en {carpeta}")
                    continue
                    
                print(f"Imágenes cargadas correctamente para {archivo} en {carpeta}")
                imagenes_filtradas.append(img_filtrada)
                imagenes_sinfondo.append(img_sinfondo)
            else:
                print(f"No se encontraron las imágenes para {archivo} en {carpeta}")
    
    return imagenes_filtradas, imagenes_sinfondo

def mostrar_imagenes(titulo, imagenes_por_verdura):
    ventana = tk.Toplevel()
    ventana.title(titulo)
    for row, imagenes in enumerate(imagenes_por_verdura):
        for col, img in enumerate(imagenes):
            img_resized = cv2.resize(img, (200, 200))
 
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
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    ventana_actual = None  # Variable para almacenar la ventana actual

    # Initialize CSV before processing begins
    crear_archivo_csv('momentos_hu2.csv', ['Nombre'] + [f'Hu{i}' for i in [0,2,4,6]] + ['Mean_B', 'Mean_G', 'Mean_R'])

    carpetas = ['berenjena', 'camote', 'papa', 'zanahoria']
    ruta_imagenes_filtradas = os.path.abspath(r'../anexos/imagenes_filtradas')
    ruta_imagenes_sinfondo = os.path.abspath(r'../anexos/imagenes_sinfondo')
    
    print(f"Ruta imágenes filtradas: {ruta_imagenes_filtradas}")
    print(f"Ruta imágenes sin fondo: {ruta_imagenes_sinfondo}")

    imagenes_filtradas, imagenes_sinfondo = obtener_imagenes_por_verdura(
        ruta_imagenes_filtradas, 
        ruta_imagenes_sinfondo, 
        carpetas
    )
    
    imagenes_por_verdura = []
    for img_filtrada, img_sinfondo, etiqueta in zip(imagenes_filtradas, imagenes_sinfondo, carpetas * (len(imagenes_filtradas) // len(carpetas))):
        if img_filtrada.size > 0 and img_sinfondo.size > 0:
            procesador = ProcesadorImagen(img_filtrada, img_sinfondo, etiqueta)
            procesador.aplicar_filtro('bordes')
            imagenes_por_verdura.append([procesador.imagen])
    
    ventana_actual = mostrar_imagenes("Imágenes Procesadas", imagenes_por_verdura)
    root.update()







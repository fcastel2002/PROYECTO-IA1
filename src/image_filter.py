import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

class ProcesadorImagen:
    def __init__(self, imagen):
        self.imagen_original = cv2.resize(imagen, (500, 500))
        self.imagen = self.imagen_original.copy()
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)


    def aplicar_filtro(self, filtro_nombre):
        if filtro_nombre == 'gaussiana':
            self.imagen = cv2.GaussianBlur(self.imagen, (7, 7), 0)
        elif filtro_nombre == 'promediada':
            self.imagen = cv2.blur(self.imagen, (2, 2))
        elif filtro_nombre == 'mediana':
            self.imagen = cv2.medianBlur(self.imagen, 15)
        elif filtro_nombre == 'bilateral':
            self.imagen = cv2.bilateralFilter(self.imagen, 11, 75, 75)
        elif filtro_nombre == 'gris':
            self.imagen = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        elif filtro_nombre == 'bordes':

            gris = self.imagen

            #_, umbral = cv2.threshold(gris, 40, 255, cv2.THRESH_BINARY)
          #  umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 6)
            # Binarización adaptativa

            # Encontrar todos los contornos

            contornos, _ = cv2.findContours(gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Encontrar el contorno más grande por área
            contorno_mas_grande = max(contornos, key=cv2.contourArea)

            # Convertir la imagen a color para dibujar contornos
            self.imagen = cv2.cvtColor(gris, cv2.COLOR_GRAY2RGB)

            # Dibujar solo el contorno más grande
            self.imagen = cv2.drawContours(self.imagen, [contorno_mas_grande], -1, (0, 255, 0), 2)


        elif filtro_nombre == 'binarizada':

            gris = self.imagen
            #_, self.imagen = cv2.threshold(gris, 130, 255, cv2.THRESH_BINARY_INV)
            self.imagen = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        elif filtro_nombre == 'gabor':
            kernel = cv2.getGaborKernel((21, 21), 8.0, 1.0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            self.imagen = cv2.filter2D(self.imagen, cv2.CV_8UC3, kernel)

        elif filtro_nombre == 'box':
            self.imagen = cv2.boxFilter(self.imagen, -1, (3, 3))

        elif filtro_nombre == 'nln':
            self.imagen = cv2.fastNlMeansDenoisingColored(self.imagen, None, 15, 15, 6, 18)

        elif filtro_nombre == 'adaptMedian':
            # Implementación de un filtro de mediana adaptativa personalizado
            self.imagen = self._filtro_mediana_adaptativa(self.imagen)

        elif filtro_nombre == 'morfologico':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            self.imagen = cv2.morphologyEx(self.imagen, cv2.MORPH_CLOSE, kernel)

        elif filtro_nombre == 'fondo':
            self.imagen = self.fgbg.apply(self.imagen)

        elif filtro_nombre == 'sobel':

            gris = self.imagen
            sobelx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=5)
            self.imagen = cv2.magnitude(sobelx, sobely).astype('uint8')

        elif filtro_nombre == 'laplaciano':

            gris = self.imagen
            self.imagen = cv2.Laplacian(gris, cv2.CV_64F).astype('uint8')

        elif filtro_nombre == 'sombras':
            hsv = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            # Procesar el canal V para minimizar sombrasc`
            v = cv2.GaussianBlur(v, (7, 7), 9)
            # Combinar los canales y convertir de nuevo a BGR
            hsv = cv2.merge([h, s, v])
            self.imagen = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif filtro_nombre == 'gamma':
            from PIL import ImageEnhance
            img_pil = Image.fromarray(cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(2)  # Ajustar gamma a 1.5
            self.imagen = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f'Filtro "{filtro_nombre}" no reconocido')

    def _filtro_mediana_adaptativa(self, imagen):
        # Filtro de mediana adaptativa: en este ejemplo se implementa una mediana con ventana adaptativa simple
        max_kernel_size = 7
        salida = imagen.copy()
        for i in range(1, max_kernel_size, 2):
            salida = cv2.medianBlur(salida, i)
        return salida

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
            if len(img_resized.shape) == 2:  # Si la imagen es en escala de grises o binarizada
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            else:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            label = Label(ventana, image=img_tk)
            label.image = img_tk
            label.grid(row=row, column=col)

if __name__ == "__main__":
    # Ruta al directorio de imágenes
    ruta_base = r'../anexos/imagenes_mias'
    carpetas = ['berenjena', 'camote', 'papa', 'zanahoria']

    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal

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
##
        primer_imagen = os.path.join(ruta_carpeta, archivos[1])
##
        imagen = cv2.imread(primer_imagen)
        if imagen is None:
            print(f'No se pudo leer la imagen {primer_imagen}.')
            continue

        # Crear instancia del procesador de imagen
        procesador = ProcesadorImagen(imagen)
        # Definir los filtros a aplicar
        filtros_a_aplicar = ['bilateral','sombras','gris','binarizada','morfologico']
        # 'mediana', 'gaussiana' 'promediada', 'bilateral', 'gris', 'bordes', 'binarizada', 'gabor', 'box', 'nln', 'adaptMedian', 'morfologico', 'sobel', 'laplaciano', 'saturacion', 'gamma'
        # Aplicar los filtros
        imagenes_filtradas = procesador.aplicar_filtros(filtros_a_aplicar)
        # Agregar las imágenes filtradas a la lista
        imagenes_por_verdura.append(imagenes_filtradas)

    # Mostrar todas las imágenes en una sola ventana
    mostrar_imagenes("Imágenes Filtradas", imagenes_por_verdura)
    root.mainloop()  # Mantener la ventana principal viva para gestionar otras ventanas


# procesador_imagen.py

import cv2
import numpy as np
from PIL import Image

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
            contornos, _ = cv2.findContours(gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contorno_mas_grande = max(contornos, key=cv2.contourArea)
            self.imagen = cv2.cvtColor(gris, cv2.COLOR_GRAY2RGB)
            self.imagen = cv2.drawContours(self.imagen, [contorno_mas_grande], -1, (0, 255, 0), 2)
        elif filtro_nombre == 'binarizada':
            gris = self.imagen
            self.imagen = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)
        elif filtro_nombre == 'gabor':
            kernel = cv2.getGaborKernel((21, 21), 8.0, 1.0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            self.imagen = cv2.filter2D(self.imagen, cv2.CV_8UC3, kernel)
        elif filtro_nombre == 'box':
            self.imagen = cv2.boxFilter(self.imagen, -1, (3, 3))
        elif filtro_nombre == 'nln':
            self.imagen = cv2.fastNlMeansDenoisingColored(self.imagen, None, 15, 5, 6, 18)
        elif filtro_nombre == 'adaptMedian':
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
            v = cv2.GaussianBlur(v, (7, 7), 1)
            hsv = cv2.merge([h, s, v])
            self.imagen = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif filtro_nombre == 'gamma':
            from PIL import ImageEnhance
            img_pil = Image.fromarray(cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(1.5)
            self.imagen = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f'Filtro "{filtro_nombre}" no reconocido')

    def _filtro_mediana_adaptativa(self, imagen):
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

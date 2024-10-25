import os
import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

class ProcesadorImagen:
    def __init__(self, imagen):
        self.imagen_original = cv2.resize(imagen, (200, 200))
        self.imagen = self.imagen_original.copy()

    def aplicar_filtro(self, filtro_nombre):
        if filtro_nombre == 'gaussiana':
            self.imagen = cv2.GaussianBlur(self.imagen, (7, 7), 0)
        elif filtro_nombre == 'promediada':
            self.imagen = cv2.blur(self.imagen, (2, 2))
        elif filtro_nombre == 'mediana':
            self.imagen = cv2.medianBlur(self.imagen, 9)
        elif filtro_nombre == 'bilateral':
            self.imagen = cv2.bilateralFilter(self.imagen, 10, 100, 75)
        elif filtro_nombre == 'gris':
            self.imagen = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        elif filtro_nombre == 'bordes':
            if len(self.imagen.shape) == 3:  # Convertir a escala de grises si es necesario
                gris = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
            else:
                gris = self.imagen
            self.imagen = cv2.Canny(gris, 80, 200)
        elif filtro_nombre == 'binarizada':
            if len(self.imagen.shape) == 3:  # Convertir a escala de grises si es necesario
                gris = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
            else:
                gris = self.imagen
            _, self.imagen = cv2.threshold(gris, 140, 255, cv2.THRESH_BINARY)
        elif filtro_nombre == 'gabor':
            kernel = cv2.getGaborKernel((21, 21), 8.0, 1.0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            self.imagen = cv2.filter2D(self.imagen, cv2.CV_8UC3, kernel)
        elif filtro_nombre == 'box':
            self.imagen = cv2.boxFilter(self.imagen, -1, (3, 3))
        elif filtro_nombre == 'nln':
            self.imagen = cv2.fastNlMeansDenoisingColored(self.imagen, None, 10, 10, 3, 14)
        elif filtro_nombre == 'adaptMedian':
            # Implementación de un filtro de mediana adaptativa personalizado
            self.imagen = self._filtro_mediana_adaptativa(self.imagen)
        elif filtro_nombre == 'morfologico':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.imagen = cv2.morphologyEx(self.imagen, cv2.MORPH_GRADIENT, kernel)
        elif filtro_nombre == 'sobel':
            if len(self.imagen.shape) == 3:  # Convertir a escala de grises si es necesario
                gris = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
            else:
                gris = self.imagen
            sobelx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=5)
            self.imagen = cv2.magnitude(sobelx, sobely).astype('uint8')
        elif filtro_nombre == 'laplaciano':
            if len(self.imagen.shape) == 3:  # Convertir a escala de grises si es necesario
                gris = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
            else:
                gris = self.imagen
            self.imagen = cv2.Laplacian(gris, cv2.CV_64F).astype('uint8')
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
            if len(img.shape) == 2:  # Si la imagen es en escala de grises o binarizada
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            label = Label(ventana, image=img_tk)
            label.image = img_tk
            label.grid(row=row, column=col)

if __name__ == "__main__":
    # Ruta al directorio de imágenes
    ruta_base = r'../anexos/imagenes_mias'
    carpetas = ['berenjena', 'camote', 'choclo', 'papa', 'zanahoria']

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

        primer_imagen = os.path.join(ruta_carpeta, archivos[15])
        imagen = cv2.imread(primer_imagen)
        if imagen is None:
            print(f'No se pudo leer la imagen {primer_imagen}.')
            continue

        # Crear instancia del procesador de imagen
        procesador = ProcesadorImagen(imagen)
        # Definir los filtros a aplicar
        filtros_a_aplicar = ['morfologico']
        # 'mediana', 'gaussiana' 'promediada', 'bilateral', 'gris', 'bordes', 'binarizada', 'gabor', 'box', 'nln', 'adaptMedian', 'morfologico', 'sobel', 'laplaciano'
        # Aplicar los filtros
        imagenes_filtradas = procesador.aplicar_filtros(filtros_a_aplicar)
        # Agregar las imágenes filtradas a la lista
        imagenes_por_verdura.append(imagenes_filtradas)

    # Mostrar todas las imágenes en una sola ventana
    mostrar_imagenes("Imágenes Filtradas", imagenes_por_verdura)

    root.mainloop()

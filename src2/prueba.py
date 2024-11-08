import os
import cv2
import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
from Archivos import *
import sys
import logging  # Add this import

# Configure logging at the beginning of the file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProcesadorImagen:
    def __init__(self):
        self.ruta_imagenes = self.obtener_ruta_imagenes()
        self.imagenes = self.cargar_imagenes(self.ruta_imagenes)
        self.indice_actual = 0
        self.ventana_principal = tk.Tk()
        self.ventana_principal.withdraw()  # Ocultar la ventana principal

    def obtener_ruta_imagenes(self):
        # Código para solicitar al usuario la ruta de las imágenes
        ruta = filedialog.askdirectory(title='Seleccione la ruta de las imágenes')
        return ruta

    def cargar_imagenes(self, ruta):
        # Código para cargar imágenes desde la ruta especificada
        imagenes = []
        for archivo in os.listdir(ruta):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                imagen = cv2.imread(os.path.join(ruta, archivo))
                if imagen is not None:
                    imagen = cv2.resize(imagen, (800, 600))  # Redimensionar a tamaño estándar
                    imagenes.append(imagen)
        return imagenes

    def procesar_siguiente_imagen(self, filtros_a_aplicar):
        if self.indice_actual >= len(self.imagenes):
            logging.info('All images have been processed.')
            return None

        try:
            imagen = self.imagenes[self.indice_actual]
            imagen_original = imagen.copy()
            imagen_procesada = imagen.copy()
            imagenes_progreso = [imagen_procesada.copy()]  # Imagen original

            for filtro in filtros_a_aplicar:
                logging.info(f'Applying filter: {filtro}')
                if filtro == 'gaussian':
                    imagen_procesada = cv2.GaussianBlur(imagen_procesada, (13,13), 0)
                elif filtro == 'mean':
                    imagen_procesada = cv2.blur(imagen_procesada, (13,13))
                elif filtro == 'gris':
                    imagen_procesada = cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2GRAY)
                elif filtro == 'binarized':
                    _, imagen_procesada = cv2.threshold(imagen_procesada, 127, 255, cv2.THRESH_BINARY)
                elif filtro == "binarizedINV":
                    _, imagen_procesada = cv2.threshold(imagen_procesada, 127, 255, cv2.THRESH_BINARY_INV)
                elif filtro == 'binarizedADAPTIVE':
                    imagen_procesada = cv2.adaptiveThreshold(imagen_procesada, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 6)
                elif filtro == 'sobel':
                    imagen_procesada = cv2.Sobel(imagen_procesada, cv2.CV_64F, 1, 0, ksize=5)
                    imagen_procesada = cv2.normalize(imagen_procesada, None, 0, 255, cv2.NORM_MINMAX)
                    imagen_procesada = np.uint8(imagen_procesada)
                elif filtro == 'laplacian':
                    imagen_procesada = cv2.Laplacian(imagen_procesada, cv2.CV_64F)
                    imagen_procesada = cv2.convertScaleAbs(imagen_procesada)
                elif filtro == 'morfologico':
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (11,11))
                    imagen_procesada = cv2.morphologyEx(imagen_procesada, cv2.MORPH_DILATE, kernel)
                    
                elif filtro == 'canny':
                    imagen_gris = imagen_procesada
                    imagen_procesada = cv2.Canny(imagen_gris, 127, 255)
                elif filtro == 'contornos':
                    contornos = cv2.findContours(imagen_procesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contornos = contornos[0] if len(contornos) == 2 else contornos[1]
                    if contornos:
                        max_contorno = max(contornos, key=cv2.contourArea)
                        imagen_procesada = imagen_original.copy()
                        cv2.drawContours(imagen_procesada, [max_contorno], -1, (0, 255, 0), 9)
                    else:
                        logging.warning("No se encontraron contornos en la imagen.")
                else:
                    logging.error(f'Filtro "{filtro}" no reconocido')
                    continue  # Skip unrecognized filters
                imagenes_progreso.append(imagen_procesada.copy())

            self.indice_actual += 1
            return imagenes_progreso
        except Exception as e:
            logging.error(f'Error processing image at index {self.indice_actual}: {e}')
            self.indice_actual += 1
            return None

    def mostrar_imagenes(self, imagenes):
        if imagenes is None:
            print("No hay más imágenes para procesar")
            return False

        # Crear una nueva ventana Toplevel para mostrar las imágenes
        ventana = tk.Toplevel(self.ventana_principal)
        ventana.title(f"Imagen {self.indice_actual} - Progreso de filtros")

        frame = tk.Frame(ventana)
        frame.pack(expand=True, fill='both', padx=10, pady=10)

        self.imagen_refs = []  # Lista para mantener referencias a las imágenes

        for idx, imagen in enumerate(imagenes):
            try:
                # Convertir y normalizar imagen
                if isinstance(imagen.dtype, np.float64):
                    imagen = cv2.normalize(imagen, None, 0, 255, cv2.NORM_MINMAX)
                    imagen = np.uint8(imagen)

                # Convertir espacio de color
                if len(imagen.shape) == 2:
                    imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
                else:
                    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

                # Redimensionar si es necesario
                altura, ancho = imagen.shape[:2]
                max_size = 400  # Tamaño más pequeño para mostrar más imágenes
                if altura > max_size or ancho > max_size:
                    ratio = min(max_size / altura, max_size / ancho)
                    nuevo_ancho = int(ancho * ratio)
                    nueva_altura = int(altura * ratio)
                    imagen = cv2.resize(imagen, (nuevo_ancho, nueva_altura))

                # Crear imagen Tkinter
                imagen_pil = Image.fromarray(imagen)
                imagen_tk = ImageTk.PhotoImage(image=imagen_pil)

                # Guardar referencias
                self.imagen_refs.append(imagen_tk)

                # Crear y posicionar label
                label = Label(frame, image=self.imagen_refs[-1])
                label.grid(row=idx//3, column=idx%3, padx=5, pady=5)

            except Exception as e:
                logging.error(f'Error displaying images: {e}')
                continue

        # Botón para cerrar la ventana
        def cerrar_ventana():
            self.ventana_principal.destroy()  # Cerrar la ventana principal
            sys.exit(0)  # Terminar el programa completamente

        btn_cerrar = tk.Button(ventana, text="Cerrar", command=cerrar_ventana)
        btn_cerrar.pack(pady=5)

        # === Cambio: Agregar botón 'Next' ===
        def siguiente_imagen():
            ventana.destroy()

        btn_next = tk.Button(ventana, text="Next", command=siguiente_imagen)
        btn_next.pack(pady=5)
        # ======================================

        # Asegurarse de que la ventana aparezca en primer plano
        ventana.focus_force()
        ventana.grab_set()
        ventana.wait_window()

        return True

if __name__ == '__main__':
    procesador = ProcesadorImagen()
    # Actualizar la lista de filtros a aplicar
    filtros_a_aplicar = ['gaussian','gris','binarizedADAPTIVE','morfologico','contornos']

    try:
        while True:
            imagenes_progreso = procesador.procesar_siguiente_imagen(filtros_a_aplicar)
            if not imagenes_progreso or not procesador.mostrar_imagenes(imagenes_progreso):
                break

            # === Cambio: Eliminar solicitud de entrada por consola ===
            # respuesta = input("¿Desea procesar la siguiente imagen? (s/n): ").lower()
            # if respuesta != 's':
            #     break
            # ======================================================

    except KeyboardInterrupt:
        logging.info("Programa terminado por el usuario")
    except Exception as e:
        logging.error(f'Error: {e}')
    finally:
        # Asegurarse de cerrar la ventana principal al finalizar
        procesador.ventana_principal.destroy()
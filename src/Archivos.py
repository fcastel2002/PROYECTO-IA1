import os
import cv2
import csv

def obtener_imagenes_por_verdura(ruta_base, carpetas, indice):
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

        imagen_path = os.path.join(ruta_carpeta, archivos[indice % len(archivos)])
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            print(f'No se pudo leer la imagen {imagen_path}.')
            continue

        imagenes_por_verdura.append(imagen)
    return imagenes_por_verdura

def crear_archivo_csv(ruta_archivo, encabezados):
    with open(ruta_archivo, 'w', newline='') as archivo_csv:
        escritor = csv.writer(archivo_csv)
        escritor.writerow(encabezados)

def agregar_fila_csv(ruta_archivo, datos):
    with open(ruta_archivo, 'a', newline='') as archivo_csv:
        escritor = csv.writer(archivo_csv)
        escritor.writerow(datos)

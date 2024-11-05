import cv2
import numpy as np
from Archivos import crear_archivo_csv, agregar_fila_csv

def calcular_momentos_hu(contorno):
    momentos = cv2.moments(contorno)
    hu_momentos = cv2.HuMoments(momentos).flatten()
    momentos_hu = [-np.sign(m)*np.log10(abs(m)) if m != 0 else 0 for m in hu_momentos.flatten()]
    
    return momentos_hu

def guardar_momentos_hu(ruta_csv, etiqueta, hu_momentos):
    datos = [etiqueta] + hu_momentos
    agregar_fila_csv(ruta_csv, datos)

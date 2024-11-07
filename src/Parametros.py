import cv2
import numpy as np
from Archivos import crear_archivo_csv, agregar_fila_csv

def calcular_momentos_hu(contorno,momentos_elegidos):
    momentos = cv2.moments(contorno)
    hu_momentos = cv2.HuMoments(momentos).flatten() 
    # Select only 2nd (index 1), 4th (index 3), and 6th (index 5) moments
    momentos_seleccionados = [hu_momentos[i] for i in momentos_elegidos]
    momentos_hu = [-np.sign(m)*np.log10(abs(m)) if m != 0 else 0 for m in momentos_seleccionados]
    
    return momentos_hu

def guardar_momentos_hu(ruta_csv, etiqueta, hu_momentos, mean_color):
    datos = [etiqueta] + hu_momentos + list(mean_color)
    agregar_fila_csv(ruta_csv, datos)

def calcular_color_promedio(imagen_sin_fondo):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(imagen_sin_fondo, cv2.COLOR_BGR2HSV)
    # Create mask to select non-black pixels
    mask = cv2.inRange(hsv_image, (0, 0, 1), (180, 255, 255))
    # Calculate the mean color using the mask
    mean_val = cv2.mean(imagen_sin_fondo, mask=mask)
    mean_color = mean_val[:3]  # B, G, R
    return mean_color
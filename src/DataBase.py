from Archivos import Archivos
from Caracteristicas import Caracteristicas
import numpy as np
import os

class DataBase:
    def __init__(self):
        self.datos = []

    def agregar_dato(self, nombre_audio, etiqueta):
        try:
            caracteristicas = Caracteristicas(nombre_audio)
            features = caracteristicas.extraer_features()
            self.datos.append(list(features) + [etiqueta])
            print(f"Audio {nombre_audio} etiquetado como {etiqueta}")
        except Exception as e:
            print(f"Error al procesar {nombre_audio}: {str(e)}")

    def guardar_base_datos(self, nombre_archivo):
        if not self.datos:
            print("No hay datos para guardar.")
            return

        # Verificar que todas las filas tengan la misma longitud
        longitud_correcta = len(self.datos[0])
        datos_filtrados = [fila for fila in self.datos if len(fila) == longitud_correcta]

        # Definir cabeceras con la longitud correcta
        cabeceras = [f"MFCC_{i}" for i in range(13)] + [f"ZCR_{i}" for i in range(10)] + ["Etiqueta"]

        # Guardar los datos filtrados en el archivo CSV
        Archivos.guardar_csv(nombre_archivo, datos_filtrados, cabeceras)
        print(f"Base de datos guardada en {nombre_archivo}")
        
        # Normalización opcional
        self.normalizar_archivo(nombre_archivo)

    def normalizar_archivo(self, archivo_origen):
        """Lee un archivo CSV, normaliza sus características y guarda un nuevo archivo normalizado"""
        datos = Archivos.leer_csv(archivo_origen)
        if not datos or len(datos) <= 1:
            print("Error: No hay suficientes datos para normalizar")
            return

        cabeceras = datos[0]
        datos_validados = [fila for fila in datos[1:] if len(fila) == len(cabeceras)]

        # Convertir las filas en características y etiquetas
        X = np.array([fila[:-1] for fila in datos_validados], dtype=float)
        y = [fila[-1] for fila in datos_validados]

        media = np.nanmean(X, axis=0)
        desv_std = np.nanstd(X, axis=0)
        desv_std[desv_std == 0] = 1  # Evita división por cero en características constantes
        
        # Normalizar las características
        X_norm = (X - media) / desv_std
        datos_normalizados = [list(fila) + [etiqueta] for fila, etiqueta in zip(X_norm, y)]
        
        archivo_normalizado = archivo_origen.replace('.csv', '_std.csv')
        Archivos.guardar_csv(archivo_normalizado, datos_normalizados, cabeceras)
        print(f"Archivo normalizado guardado como {archivo_normalizado}")

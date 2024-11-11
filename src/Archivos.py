import csv
import os

class Archivos:
    @staticmethod
    def guardar_csv(nombre_archivo, datos, cabeceras=None):
        modo = 'a' if os.path.exists(nombre_archivo) else 'w'
        try:
            with open(nombre_archivo, mode=modo, newline='') as archivo_csv:
                escritor = csv.writer(archivo_csv)
                if cabeceras and modo == 'w':
                    escritor.writerow(cabeceras)
                escritor.writerows(datos)
            print(f"Datos guardados en {nombre_archivo}.")
        except Exception as e:
            print(f"Error al guardar datos en {nombre_archivo}: {e}")

    @staticmethod
    def leer_csv(nombre_archivo):
        try:
            with open(nombre_archivo, mode='r') as archivo_csv:
                lector = csv.reader(archivo_csv)
                datos = [fila for fila in lector]
            print(f"Datos leídos de {nombre_archivo}.")
            return datos
        except FileNotFoundError:
            print(f"Error: El archivo '{nombre_archivo}' no existe.")
            return []

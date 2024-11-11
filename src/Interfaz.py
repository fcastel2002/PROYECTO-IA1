from DataBase import *
from Caracteristicas import Caracteristicas
from AnalisisDimensional import *
import os
import keyboard
import time

class Interfaz:
    def __init__(self):
        self.database = DataBase()

    def menu_principal(self):
        while True:
            print("\n--- Menú Principal ---")
            print("1. Filtrar y etiquetar audios de carpeta")
            print("2. Realizar análisis PCA")
            print("3. Salir")
            opcion = input("Seleccione una opción: ")

            if opcion == '1':
                self.filtrar_y_etiquetar_audios()
            elif opcion == '2':
                self.realizar_pca()
            elif opcion == '3':
                break
            else:
                print("Opción inválida. Intente nuevamente.")

    def filtrar_y_etiquetar_audios(self):
        filtered_dir = '../anexos/audios_filtered'
        os.makedirs(filtered_dir, exist_ok=True)
        
        for filename in os.listdir(filtered_dir):
            if filename.endswith('.wav'):
                nombre_audio = filename[:-4]
                etiquetas = {'1': 'papa', '2': 'zanahoria', '3': 'camote', '4': 'berenjena'}
                while True:
                    print(f"\nAudio a etiquetar: {nombre_audio}")
                    for key, value in etiquetas.items():
                        print(f"{key}. {value}")
                    opcion = input("Seleccione una etiqueta: ").strip()
                    if opcion in etiquetas:
                        etiqueta = etiquetas[opcion]
                        self.database.agregar_dato('../anexos/audios_filtered/'+nombre_audio, etiqueta)
                        break
                    else:
                        print("Opción inválida. Intente nuevamente.")
        
        self.database.guardar_base_datos("base_datos.csv")

    def realizar_pca(self):
        analisis = AnalisisDimensional("base_datos.csv")
        componentes = analisis.aplicar_pca()
        if componentes is not None:
            analisis.graficar_componentes(componentes)

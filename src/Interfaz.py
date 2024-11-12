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
            print("1. Filtrar audios")
            print("2. Filtrar y etiquetar audios de carpeta")
            print("3. Realizar análisis PCA")
            print("4. Salir")
            opcion = input("Seleccione una opción: ")

            if opcion == '1':
                self.extraer_caracteristicas_audios()
            elif opcion == '2':
                self.etiquetar_audios()
            elif opcion == '3':
                self.realizar_pca()
            elif opcion == '4':
                break
            else:
                print("Opción inválida. Intente nuevamente.")

    def extraer_caracteristicas_audios(self):
        """Extrae características de los audios, guarda audios filtrados y las características en un archivo CSV con etiquetas."""
        raw_dir = '../anexos/audios_raw'
        filtered_dir = '../anexos/audios_filtered'
        os.makedirs(filtered_dir, exist_ok=True)
        
        for filename in os.listdir(raw_dir):
            if filename.endswith('.wav'):
                nombre_audio = os.path.join(raw_dir, filename)
                try:
                    print(f"Procesando: {filename}")
                    # Extraer características
                    caracteristicas = Caracteristicas(nombre_audio[:-4])  # Sin extensión
                    features = caracteristicas.extraer_features()
                    
                    # Guardar audio filtrado con el mismo nombre
                    filtered_audio_path = os.path.join(filtered_dir, filename)
                    caracteristicas.guardar_audio_filtrado(filtered_audio_path)
                    
                    # Obtener etiqueta del nombre del archivo
                    etiqueta = filename.split('_')[0]
                    
                    # Añadir las características y etiqueta a la base de datos
                    self.database.agregar_dato(nombre_audio[:-4], etiqueta)
                    
                except Exception as e:
                    print(f"Error al procesar {filename}: {e}")
        
        # Guardar los datos en el archivo CSV utilizando DataBase
        self.database.guardar_base_datos("base_datos.csv")
        print("Características guardadas en base_datos.csv")

    def etiquetar_audios(self):
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
        analisis = AnalisisDimensional("base_datos_std.csv")
        componentes = analisis.aplicar_pca()
        if componentes is not None:
            analisis.graficar_componentes(componentes)

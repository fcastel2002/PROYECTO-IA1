import csv
import os
import pandas as pd
import numpy as np
from Caracteristicas import Caracteristicas
import soundfile as sf

class DataBase:
    
    def __init__(self, carpeta_prefiltrados):
        self.carpeta_prefiltrados = '../anexos/test_filt'
        self.csv_file = 'caracteristicas.csv' 
        self.csv_file_std = 'caracteristicas_std.csv'
        self.csv_file_normalized = 'caracteristicas_normalized.csv'
        self.header = []

    def get_header(self, segmentos, n_mfcc, n_formantes):
        for i in range(segmentos):
            #self.header.append(f'zcr_{i}')

            for j in range(n_mfcc):
                self.header.append(f'mfcc_{i}_{j}')
        for j in range(n_formantes):
            self.header.append(f'formantes_{j}')
            self.header.append(f'amp_formantes_{j}')
        self.header.append('spc_bw')
        self.header.append('flatness')
        self.header.append('Etiqueta')

    def extraer_caracteristicas(self):
        flag = False
        n_segmentos = 6
        n_mfcc = 13
        n_formantes = 3
        for root, dirs, files in os.walk(self.carpeta_prefiltrados):
            for file in files:
                if file.endswith(".wav"):
                    print(f"Extrayendo características de {file}")
                    ruta = os.path.join(root, file)
                    
                    # Crear instancia de Caracteristicas
                   
                    
                    caracteristicas = Caracteristicas(ruta, n_segmentos=n_segmentos, n_mfcc=n_mfcc, n_formantes=n_formantes)  # Ajusta parámetros según necesidad
                    features = caracteristicas.extraer()
                    
                    etiqueta = file.split("_")[0]
                    self.guardar_en_csv(features, etiqueta, flag,n_segmentos,n_mfcc,n_formantes)
                    flag = True

    def guardar_en_csv(self, features, etiqueta, flag,n_s,n_m,n_f):
        try:
            mode = 'a' if flag else 'w'
            with open(self.csv_file, mode=mode, newline='') as file:
                writer = csv.writer(file)
                if mode == 'w':
                    self.get_header(segmentos=n_s, n_mfcc=n_m, n_formantes=n_f)  # Ajusta parámetros según necesidad
                    writer.writerow(self.header)
                features = list(features)
                features.append(etiqueta)
                writer.writerow(features)
        except Exception as e:
            print("Error al guardar en CSV: ", e)

    def estandarizar_database(self):
        try:
            if os.path.exists(self.csv_file_std):
                os.remove(self.csv_file_std)
            
            # Cargar base de datos original
            data = pd.read_csv(self.csv_file)
            etiquetas = data.pop('Etiqueta')
            
            # Estandarizar características
            data = (data - data.mean()) / data.std()
            
            # Agregar etiquetas nuevamente
            data['Etiqueta'] = etiquetas
            
            # Guardar archivo estandarizado
            with open(self.csv_file_std, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.header)
                writer.writerows(data.values)
        except Exception as e:
            print("Error al estandarizar la base de datos: ", e)

    def normalizar_database(self):
        try:
            csv_file_normalized = 'caracteristicas_normalized.csv'
            if os.path.exists(csv_file_normalized):
                os.remove(csv_file_normalized)
            
            # Cargar base de datos original
            data = pd.read_csv(self.csv_file)
            etiquetas = data.pop('Etiqueta')
            
            # Normalizar características
            data = (data - data.min()) / (data.max() - data.min())
            
            # Agregar etiquetas nuevamente
            data['Etiqueta'] = etiquetas
            
            # Guardar archivo normalizado
            with open(csv_file_normalized, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.header)
                writer.writerows(data.values)
        except Exception as e:
            print("Error al normalizar la base de datos: ", e)
from prediccion import *
from DataBase2 import *
from Filtrado2 import *
from KNNPredictor import *
from AudiosRaw import *
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk, simpledialog
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from PIL import Image, ImageTk  # Added import for image handling
import os
import shutil  # Added imports for file operations
from PCA import *
import pickle
from server import *
import threading


def run_flask_server():
    # Ejecutar el servidor Flask
    app.run(host='0.0.0.0', port=5000)


class UserInterface:

    def __init__(self, carpeta_crudos, carpeta_prefiltrados):
        flask_thread = threading.Thread(target=run_flask_server)
        flask_thread.daemon = True  # El hilo se cerrará cuando el programa principal termine
        flask_thread.start()

        self.carpeta_crudos = carpeta_crudos
        self.carpeta_prefiltrados = carpeta_prefiltrados
        self.database = DataBase(
            self.carpeta_prefiltrados, 'audios_features.csv')
        self.filtrado = Filtrado(
            self.carpeta_crudos, self.carpeta_prefiltrados)
        self.grabacion = AudiosRaw(self.carpeta_crudos)

        self.resultados_prediccion = []  # Store prediction results
        self.resultados_imagenes = []  # Store image prediction results

        self.window = tk.Tk()
        self.window.title("Interfaz de usuario")
        self.window.geometry("400x400")
        self.window.resizable(False, False)
        self.create_widgets()
        self.window.mainloop()

    def create_widgets(self):
        self.label = tk.Label(self.window, text="Seleccione una opción:")
        self.label.pack(pady=10)

        self.button5 = ttk.Button(
            self.window, text="Iniciar grabación", command=self.iniciar_grabacion)
        self.button5.pack(pady=10)

        self.button1 = ttk.Button(
            self.window, text="Filtrar y normalizar audios", command=self.filtrado.procesar_audios)
        self.button1.pack(pady=10)

        self.button2 = ttk.Button(
            self.window, text="Extraer características", command=self.database.extraer_caracteristicas)
        self.button2.pack(pady=10)

        self.button3 = ttk.Button(self.window, text="Predecir con KNN")
        self.button3.pack(pady=10)
        self.button3.config(command=self.ingresar_valor_k)

        self.button6 = ttk.Button(
            self.window, text="Predecir imagenes", command=self.ejecutar_prediccion_imagenes)
        self.button6.pack(pady=10)

        self.button4 = ttk.Button(
            self.window, text="Salir", command=self.window.quit)
        self.button4.pack(pady=10)

    def iniciar_grabacion(self):
        self.borrar_archivos()
        self.grabacion.iniciar()

    def borrar_archivos(self):
        carpetas = ["../anexos/predict_audios",
                    "../anexos/predict_audios_filt"]
        for carpeta in carpetas:
            if os.path.exists(carpeta):
                shutil.rmtree(carpeta)
                os.makedirs(carpeta)

    def prediccion_knn(self, k):
        self.knn = KNN(k)
        features, etiquetas = self.estandarizar_db()
        # Asegurar arrays numpy
        self.knn.ajustar(features.values, etiquetas.values)
        puntos, labels = self.estandarizar_datos()  # Nuevos puntos a predecir
        puntos = puntos.values  # Convertimos a array numpy
        with open('scaler_umap.pkl', 'rb') as file:
            scaler = pickle.load(file)
        puntos = scaler.transform(puntos)

        print(f"Puntos shape: {puntos.shape}")

        # Cargar modelo PCA
        with open("umap_model.pkl", "rb") as file:
            pca = pickle.load(file)

        # Transformar los puntos con PCA
        puntos_reducidos = pca.transform(puntos)
        print(f"Puntos reducidos shape: {puntos_reducidos.shape}")

        # Predecir etiquetas para todos los puntos reducidos
        predicciones = self.knn.predecir(puntos_reducidos)

        # Mostrar resultados
        self.resultados_prediccion = list(zip(labels, predicciones))
        resultado = "\n".join([f"Etiqueta: {label}, Predicción: {
            pred}" for label, pred in zip(labels, predicciones)])
        messagebox.showinfo("Predicción KNN", f"Predicciones:\n{resultado}")

        # Mostrar imágenes correspondientes a las predicciones
        try:
            for label in predicciones:
                imagen_path = '../anexos/imagenes_correctas/tests/' + \
                    self.obtener_imagen_por_etiqueta(label)
                if imagen_path:
                    self.mostrar_imagen(imagen_path)
        except Exception as e:
            print(f"Error al mostrar imagen: {e}")

    def obtener_imagen_por_etiqueta(self, etiqueta):
        # Buscar la ruta de la imagen correspondiente a la etiqueta
        for archivo, label in self.resultados_imagenes:
            if label == etiqueta:
                return archivo  # Asume que 'archivo' es la ruta de la imagen
        return None

    def mostrar_imagen(self, path):
        ventana_imagen = tk.Toplevel(self.window)
        ventana_imagen.title(f"Imagen para {path}")
        img = Image.open(path)
        img = img.resize((200, 200))  # Ajusta el tamaño según sea necesario
        photo = ImageTk.PhotoImage(img)
        label_img = tk.Label(ventana_imagen, image=photo)
        label_img.image = photo  # Mantener una referencia
        label_img.pack()

    def estandarizar_datos(self):
        df = pd.read_csv("audios_features.csv")
        etiquetas = df["Etiqueta"]
        features = df.drop(columns=["Etiqueta"])

        return features, etiquetas

    def estandarizar_db(self):
        df = pd.read_csv("FINAL_DB.csv")
        etiquetas = df["Etiqueta"]
        features = df.drop(columns=["Etiqueta"])
        return features, etiquetas

    def ingresar_valor_k(self):
        k_value = tk.simpledialog.askinteger(
            "Input", "Ingrese el valor de K:", minvalue=1, maxvalue=21)
        if k_value is not None:
            self.prediccion_knn(k_value)

    def ejecutar_prediccion_imagenes(self):
        analisis = Analisis()
        analisis.run()
        self.resultados_imagenes = analisis.resultados_prediccion
        # Optionally, display or process the results
        resultado = "\n".join([f"Archivo: {archivo}, Clasificación: {etiqueta}"
                              for archivo, etiqueta in self.resultados_imagenes])
        messagebox.showinfo("Predicción de Imágenes",
                            f"Resultados:\n{resultado}")


interface = UserInterface("../anexos/predict_audios",
                          "../anexos/predict_audios_filt")

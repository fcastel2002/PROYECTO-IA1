import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2
from Parametros import calcular_momentos_hu, calcular_color_promedio, guardar_momentos_hu
import subprocess
import os
# Add imports for PIL and Tkinter image display
from PIL import Image, ImageTk
import tkinter as tk


def aumentar_saturacion_brillo(imagen, mask, factor_saturacion=1.03, factor_brillo=1.7):
    # Convertir la imagen a espacio de color HSV
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Aumentar la saturación y el brillo solo en la región del contorno
    imagen_hsv[:, :, 1] = np.where(mask == 255, np.clip(
        imagen_hsv[:, :, 1] * factor_saturacion, 0, 255), imagen_hsv[:, :, 1])
    imagen_hsv[:, :, 2] = np.where(mask == 255, np.clip(
        imagen_hsv[:, :, 2] * factor_brillo, 0, 255), imagen_hsv[:, :, 2])

    # Convertir de nuevo a espacio de color BGR
    imagen_bgr = cv2.cvtColor(imagen_hsv, cv2.COLOR_HSV2BGR)
    return imagen_bgr


def calcular_color_promedio_normalizado(imagen, mask):
    # Normalizar la imagen
    imagen_normalizada = cv2.normalize(imagen, None, 0, 255, cv2.NORM_MINMAX)
    return calcular_color_promedio(imagen_normalizada, mask)


class Analisis:
    def __init__(self):
        self.resultados_prediccion = []  # Store prediction results

    def seleccionar_carpeta(self):
        try:
            # Intentar usar zenity para seleccionar carpeta
            cmd = ['zenity', '--file-selection',
                   '--directory',
                   '--title=Seleccione la carpeta con imágenes',
                   '--filename=' + os.path.expanduser('../anexos/imagenes_mias')]

            resultado = subprocess.run(cmd, capture_output=True, text=True)

            if (resultado.returncode == 0):
                return resultado.stdout.strip()

            return self._usar_tkinter_fallback_carpeta()

        except FileNotFoundError:
            return self._usar_tkinter_fallback_carpeta()

    def _usar_tkinter_fallback_carpeta(self):
        ventana = tk.Tk()
        ventana.withdraw()
        carpeta = filedialog.askdirectory(
            initialdir='../anexos/imagenes_correctas',
            title='Seleccione la carpeta con imágenes'
        )
        ventana.destroy()
        return carpeta

    def procesar_carpeta(self, ruta_carpeta, filtros_a_aplicar, momentos_elegidos, ruta_csv):
        resultados = []
        for archivo in os.listdir(ruta_carpeta):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_completa = os.path.join(ruta_carpeta, archivo)
                features = self.procesar_imagen(
                    ruta_completa, filtros_a_aplicar, momentos_elegidos, ruta_csv)
                if features is not None:
                    resultados.append((archivo, features))
        return resultados

    def seleccionar_imagen(self):
        try:
            # Intentar usar zenity (para GNOME)
            cmd = ['zenity', '--file-selection',
                   '--title=Seleccione la imagen a analizar',
                   '--file-filter=*.png *.jpg *.jpeg',
                   '--filename=' + os.path.expanduser('../anexos/imagenes_mias')]

            resultado = subprocess.run(cmd, capture_output=True, text=True)

            if resultado.returncode == 0:
                return resultado.stdout.strip()

            # Si zenity falla, intentar kdialog (para KDE)
            cmd = ['kdialog', '--getopenfilename',
                   os.path.expanduser('~/Pictures'),
                   'Image Files (*.png *.jpg *.jpeg)']

            resultado = subprocess.run(cmd, capture_output=True, text=True)

            if resultado.returncode == 0:
                return resultado.stdout.strip()

            # Si ambos fallan, usar el diálogo tk por defecto
            return self._usar_tkinter_fallback()

        except FileNotFoundError:
            # Si no se encuentra zenity ni kdialog, usar tkinter
            return self._usar_tkinter_fallback()

    def _usar_tkinter_fallback(self):
        ventana = tk.Tk()
        ventana.withdraw()
        archivo = filedialog.askopenfilename(
            initialdir='../anexos/imagenes_correctas',
            title='Seleccione la imagen a analizar',
            filetypes=[('Image Files', '*.png *.jpg *.jpeg')]
        )
        ventana.destroy()
        return archivo

    def procesar_imagen(self, ruta_imagen, filtros_a_aplicar, momentos_elegidos, ruta_csv):
        import logging
        # Configurar logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s')
        try:
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                logging.error(f"No se pudo leer la imagen: {ruta_imagen}")
                return None
            imagen_original = imagen.copy()
            imagen_procesada = imagen.copy()

            for filtro in filtros_a_aplicar:
                logging.info(f'Aplicando filtro: {filtro}')
                if filtro == 'gaussian':
                    imagen_procesada = cv2.GaussianBlur(
                        imagen_procesada, (13, 13), 0)
                elif filtro == 'gris':
                    imagen_procesada = cv2.cvtColor(
                        imagen_procesada, cv2.COLOR_BGR2GRAY)
                elif filtro == 'binarizedADAPTIVE':
                    imagen_procesada = cv2.adaptiveThreshold(
                        imagen_procesada, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 6)
                elif filtro == 'morfologico':
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_CROSS, (11, 11))
                    imagen_procesada = cv2.morphologyEx(
                        imagen_procesada, cv2.MORPH_DILATE, kernel)
                elif filtro == 'contornos':
                    contornos = cv2.findContours(
                        imagen_procesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contornos = contornos[0] if len(
                        contornos) == 2 else contornos[1]
                    if contornos:
                        max_contorno = max(contornos, key=cv2.contourArea)
                        mask = np.zeros(
                            imagen_original.shape[:2], dtype=np.uint8)
                        cv2.drawContours(
                            mask, [max_contorno], -1, 255, thickness=cv2.FILLED)
                        # Calcular momentos de Hu y color promedio
                        hu_momentos = calcular_momentos_hu(
                            max_contorno, momentos_elegidos)
                        imagen_pospro = aumentar_saturacion_brillo(
                            imagen_original, mask)
                        mean_color = calcular_color_promedio(
                            imagen_pospro, mask)
                        # Guardar resultados en CSV
                        encabezado = [
                            'Nombre'] + [f'Hu{i+1}' for i in momentos_elegidos] + ['Mean_B', 'Mean_G', 'Mean_R']
                        guardar_momentos_hu(
                            ruta_csv, 'Imagen', hu_momentos, mean_color, encabezado)
                        # Retornar características para predicción
                        root = tk.Toplevel()
                        root.title('Imagen Filtrada')
                        imagen_rgb = cv2.cvtColor(
                            imagen_procesada, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(imagen_rgb)
                        img_pil = img_pil.resize(
                            (800, 600), Image.LANCZOS)  # Resize the image
                        img_tk = ImageTk.PhotoImage(img_pil)
                        root.geometry("800x600")  # Resize the window
                        label = tk.Label(root, image=img_tk)
                        label.image = img_tk  # Keep a reference to avoid garbage collection
                        label.pack()
                        root.wait_window(root)  # Wait for the window to close
                        return np.array(hu_momentos + list(mean_color))
                    else:
                        logging.warning(
                            "No se encontraron contornos en la imagen.")
                        return None
                else:
                    logging.error(f'Filtro "{filtro}" no reconocido')

        except Exception as e:
            logging.error(f'Error al procesar la imagen: {e}')
            return None

    def mostrar_resultados(self, resultados, predictor):
        etiquetas = ['papa', 'berenjena', 'camote', 'zanahoria']
        self.resultados_prediccion = []  # Clear previous results

        for archivo, features in resultados:
            cluster = predictor.predecir_cluster(features)
            if cluster is not None:
                etiqueta = etiquetas[cluster]
                self.resultados_prediccion.append((archivo, etiqueta))
                # ...existing code...

        # Optionally, print or log the results
        print("\nResultados de la clasificación:")
        print("="*50)
        print(f"{'Nombre de archivo':<30} {'Clasificación':<20}")
        print("-"*50)

        clasificaciones = {}
        for archivo, features in resultados:
            cluster = predictor.predecir_cluster(features)
            if cluster is not None:
                etiqueta = etiquetas[cluster]
                print(f"{archivo:<30} {etiqueta:<20}")
                if etiqueta not in clasificaciones:
                    clasificaciones[etiqueta] = 0
                clasificaciones[etiqueta] += 1

        print("\nResumen:")
        print("="*30)
        for etiqueta, cantidad in clasificaciones.items():
            print(f"{etiqueta}: {cantidad} imágenes")

    def run(self):
        ruta_carpeta = self.seleccionar_carpeta()
        if not ruta_carpeta:
            print("No se seleccionó ninguna carpeta.")
            return

        filtros_a_aplicar = ['gaussian', 'gris',
                             'binarizedADAPTIVE', 'morfologico', 'contornos']
        momentos_elegidos = [1, 2]
        ruta_csv = 'predicciones.csv'

        resultados = self.procesar_carpeta(
            ruta_carpeta, filtros_a_aplicar, momentos_elegidos, ruta_csv)
        if not resultados:
            print("No se encontraron imágenes para procesar en la carpeta.")
            return

        predictor = Predictor()
        self.mostrar_resultados(resultados, predictor)


class Estandarizacion:
    def __init__(self, features_df):
        self.mean = features_df.mean()
        self.std = features_df.std()

    def estandarizar(self, features):
        return (features - self.mean) / self.std


class Predictor:
    def __init__(self):
        self.centroides_df = pd.read_csv('centroides_normalizados.csv')
        # Omitir columna 'Cluster'
        self.centroides = self.centroides_df.iloc[:, 1:].values
        self.clusters = self.centroides_df['Cluster'].values

    def predecir_cluster(self, features):
        try:
            # Leer las características para calcular la media y desviación estándar
            features_df = pd.DataFrame([features], columns=[f'Feature_{
                                       i+1}' for i in range(len(features))])
            estandarizador = Estandarizacion(features)

            # Estandarizar las características
            features_estandarizadas = estandarizador.estandarizar(features)

            # Guardar las características estandarizadas en un archivo CSV
            features_estandarizadas_df = pd.DataFrame([features_estandarizadas], columns=[
                                                      f'Feature_{i+1}' for i in range(len(features_estandarizadas))])
            features_estandarizadas_df.to_csv(
                'clustering_std.csv', mode='a', header=not os.path.exists('clustering_std.csv'), index=False)

            # Calcular distancias y encontrar el clúster más cercano
            distances = np.linalg.norm(
                self.centroides - features_estandarizadas, axis=1)
            idx_min = np.argmin(distances)
            return self.clusters[idx_min]
        except Exception as e:
            print(f"Error al predecir clúster: {e}")
            return None


if __name__ == '__main__':
    analisis = Analisis()
    analisis.run()

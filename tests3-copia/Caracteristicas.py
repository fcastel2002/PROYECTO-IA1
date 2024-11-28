import numpy as np
import librosa as lb
import os
from scipy.signal import find_peaks
from scipy.fft import fft
import soundfile as sf  # Add this import

class Caracteristicas:
    
    def __init__(self, ruta, n_segmentos=6, n_mfcc=8, n_formantes=5):
        self.ruta = ruta
        self.audio, self.fs = lb.load(ruta, sr=None)  # Cargar audio ya filtrado y normalizado
        self.audio_name = os.path.basename(ruta)
        
        self.n_segmentos = n_segmentos
        self.n_mfcc = n_mfcc
        self.n_formantes = n_formantes
        
        # Inicializar variables de características
        self.segmentos = None
        self.zcr_ = np.array([])
        self.mfcc_ = np.array([])
        self.formantes_ = np.array([])
        self.amp_formantes_ = np.array([])
        self.spectral_bandwidth_ = np.array([])
        self.flatness_ = np.array([])
        
    def recortar_audio(self):
        try:
            umbral_energia = np.percentile(np.abs(self.audio), 25)
            len_min_palabra = int(0.05 * self.fs)
            
            segmentos_alta_energia = []
            segmento_actual = []

            for sample in self.audio:
                if np.abs(sample) > umbral_energia:
                    segmento_actual.append(sample)
                elif len(segmento_actual) >= len_min_palabra:
                    segmentos_alta_energia.append(segmento_actual)
                    segmento_actual = []

            # Agregar el último segmento si existe
            if len(segmento_actual) >= len_min_palabra:
                segmentos_alta_energia.append(segmento_actual)

            # Concatenar los segmentos de alta energía
            if segmentos_alta_energia:
                self.audio = np.concatenate(segmentos_alta_energia)
            else:
                print("Advertencia: No se encontraron segmentos de alta energía.")

            # Guardar el audio recortado
            self.guardar_audio_recortado()
        except Exception as e:
            print(f"Error al recortar audio: {e}")

    def guardar_audio_recortado(self, output_folder="../anexos/test_recortados"):
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path = os.path.join(output_folder, self.audio_name)
            sf.write(output_path, self.audio, self.fs)
            #print(f"Audio recortado guardado en: {output_path}")
        except Exception as e:
            print(f"Error al guardar audio recortado: {e}")

    def segmentar_audio(self):
        try:
            self.segmentos = [np.array(segment) for segment in np.array_split(self.audio, self.n_segmentos)]
        except Exception as e:
            print(f"Error al segmentar audio: {e}")

    def get_zero_crossing_rate(self):
        try:
            self.zcr_ = np.array([np.mean(lb.feature.zero_crossing_rate(segmento)) for segmento in self.segmentos])
        except Exception as e:
            print(f"Error al calcular zero crossing rate: {e}")

    def calcular_mfcc(self):
        try:
            mfcc_values = [np.mean(lb.feature.mfcc(y=segmento, sr=self.fs, n_mfcc=self.n_mfcc), axis=1) for segmento in self.segmentos]
            self.mfcc_ = np.concatenate(mfcc_values)
            #print(f"MFCC calculado")
        except Exception as e:
            print(f"Error al calcular MFCC: {e}")

    def get_formantes(self):
        try:
            n_coeff = 2 * self.n_formantes + 2
            lpc_coeffs = lb.lpc(self.audio, order=n_coeff)
            
            # Respuesta en frecuencia de los coeficientes LPC
            freqs = np.linspace(0, self.fs / 2, len(self.audio) // 2)
            response = np.abs(fft(lpc_coeffs, n=len(self.audio) // 2))
            
            # Encuentra picos en la respuesta (formantes)
            peaks, _ = find_peaks(response)
            self.formantes_ = freqs[peaks][:self.n_formantes]
            self.amplitudes_formantes_ = response[peaks][:self.n_formantes]  # Amplitudes de los formantes
        except Exception as e:
            print(f"Error al calcular formantes: {e}")
            self.formantes_ = np.array([])
            self.amplitudes_formantes_ = np.array([])
            
    def calcular_spectral_bandwidth(self):
        try:
            self.spectral_bandwidth_ = np.mean(lb.feature.spectral_bandwidth(y=self.audio, sr=self.fs))
        except Exception as e:
            print(f"Error al calcular spectral bandwidth: {e}")
            self.spectral_bandwidth_ = 0
   
    def calcular_flatness(self):
        try:
            self.flatness_ = np.mean(lb.feature.spectral_flatness(y=self.audio))
        except Exception as e:
            print(f"Error al calcular spectral flatness: {e}")
            self.flatness_ = 0

    def extraer(self):
        try:
            self.recortar_audio()
            self.segmentar_audio()
            self.get_zero_crossing_rate()
            self.calcular_mfcc()
            self.get_formantes()
            self.calcular_spectral_bandwidth()
            self.calcular_flatness()
            return np.concatenate([self.mfcc_, self.formantes_, self.amplitudes_formantes_, [self.spectral_bandwidth_], [self.flatness_]])
        
        except Exception as e:
            print(f"Error al extraer características: {e}")
            return None
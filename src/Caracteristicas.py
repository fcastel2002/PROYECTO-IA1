import numpy as np
import librosa

class Caracteristicas:
    def __init__(self, ruta_audio):
        self.ruta_audio = ruta_audio + '.wav'
        self.audio = None
        self.frecuencia_muestreo = None

    def cargar_audio(self):
        self.audio, self.frecuencia_muestreo = librosa.load(self.ruta_audio, duration=1.0)
        print("Audio cargado y recortado a 1 segundo.")

    def aplicar_filtro(self):
        if self.audio is not None and len(self.audio) > 0:
            audio_filtrado_bajo = librosa.effects.preemphasis(self.audio, coef=0.97)
            audio_filtrado_alto = librosa.effects.preemphasis(audio_filtrado_bajo, coef=-0.97)
            self.audio = audio_filtrado_alto / np.max(np.abs(audio_filtrado_alto))
            print("Filtros aplicados y volumen normalizado.")

    def recortar_por_volumen(self, umbral=0.05):
        if self.audio is not None and len(self.audio) > 0:
            top_db = -20 * np.log10(umbral)
            indices = librosa.effects.trim(self.audio, top_db=top_db)[1]
            self.audio = self.audio[indices[0]:indices[1]]
            print("Audio recortado para mantener volumen considerable.")

    def reducir_ruido(self, ruido=None):
        if self.audio is not None and len(self.audio) > 0:
            if ruido is None:
                ruido = self.audio[:int(0.5 * self.frecuencia_muestreo)]
            ruido_medio = np.mean(ruido)
            self.audio = self.audio - ruido_medio
            print("Reducción de ruido aplicada.")

    def dividir_y_calcular_zcr(self, segmentos=10):
        if self.audio is None or len(self.audio) == 0:
            return [0] * segmentos  # Devuelve un vector de ceros en lugar de ZCR

        duracion_segmento = int(self.frecuencia_muestreo * 0.1)  # 0.1 segundos por segmento
        zcr_valores = []

        for i in range(segmentos):
            inicio = i * duracion_segmento
            fin = inicio + duracion_segmento
            segmento = self.audio[inicio:fin]
            
            if len(segmento) > 0:
                zcr = librosa.feature.zero_crossing_rate(segmento)[0]
                zcr_valores.append(np.mean(zcr))
            else:
                zcr_valores.append(0)

        print("Zero-Crossing Rate calculado para cada segmento de 0.1 segundos.")
        return zcr_valores

    def extraer_mfcc(self, n_mfcc=13):
        if self.audio is not None and len(self.audio) > 0:
            mfcc = librosa.feature.mfcc(y=self.audio, sr=self.frecuencia_muestreo, n_mfcc=n_mfcc)
            print("Características MFCC extraídas.")
            return mfcc.flatten()[:n_mfcc]  # Limitar a los primeros 13 coeficientes MFCC
        else:
            return np.zeros(n_mfcc)

    def extraer_features(self):
        self.cargar_audio()
        self.aplicar_filtro()
        self.recortar_por_volumen()
        self.reducir_ruido()

        # Extracción de características individuales
        mfcc_features = self.extraer_mfcc()
        zcr_features = self.dividir_y_calcular_zcr()

        # Concatenación final asegurada en tamaño fijo
        self.caracteristicas = np.concatenate((mfcc_features, zcr_features))
        print(f"Características extraídas: {self.caracteristicas}")
        return self.caracteristicas

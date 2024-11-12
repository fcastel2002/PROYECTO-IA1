import numpy as np
import librosa
import soundfile as sf  # Asegúrate de tener instalado la biblioteca soundfile
import pyworld as pw
class Caracteristicas:
    def __init__(self, ruta_audio):
        self.ruta_audio = ruta_audio + '.wav'
        self.audio = None
        self.frecuencia_muestreo = None
        self.f0_deseada = 300
    def cargar_audio(self):
        self.audio, self.frecuencia_muestreo = librosa.load(self.ruta_audio, duration=1.5)
        print("Audio cargado y recortado a 1 segundo.")

    def aplicar_filtro(self):
        if self.audio is not None and len(self.audio) > 0:
            audio_filtrado_bajo = librosa.effects.preemphasis(self.audio, coef=0.97)
            audio_filtrado_alto = librosa.effects.preemphasis(audio_filtrado_bajo, coef=-0.97)
            # RMS normalization
            rms = np.sqrt(np.mean(audio_filtrado_alto**2))
            if rms > 0:
                self.audio = audio_filtrado_alto / rms / 3
            else:
                self.audio = audio_filtrado_alto
            print("Filtros aplicados y volumen normalizado con RMS.")

    def recortar_por_volumen(self, umbral=20, margen_ms=70, min_duracion_ms=100, silencio_transicion_ms=20):
        if self.audio is not None and len(self.audio) > 0:
            # Duración del margen en muestras
            margen = int(self.frecuencia_muestreo * (margen_ms / 1000))
            min_duracion = int(self.frecuencia_muestreo * (min_duracion_ms / 1000))
            silencio_transicion = np.zeros(int(self.frecuencia_muestreo * (silencio_transicion_ms / 1000)))

            # Realizar el split con el umbral ajustado de top_db
            intervals = librosa.effects.split(self.audio, top_db=umbral)

            # Crear una lista para almacenar los segmentos con márgenes añadidos
            recorte_audio = []
            for start, end in intervals:
                # Filtrar intervalos demasiado cortos para evitar cortes de "tartamudeo"
                if end - start >= min_duracion:
                    # Expande los intervalos en ambas direcciones para evitar cortes en palabras lentas
                    start = max(0, start - margen)
                    end = min(len(self.audio), end + margen)
                    recorte_audio.append(self.audio[start:end])
                    # Añadir una transición de silencio entre intervalos para suavizar la unión
                    recorte_audio.append(silencio_transicion)

            # Concatenar todos los intervalos en una sola señal continua
            self.audio = np.concatenate(recorte_audio)
            print("Audio recortado con márgenes y transición de silencio para suavizar.")

    def detectar_f0(self):
        """Detecta la frecuencia fundamental (f0) del audio usando el método YIN."""
        f0 = librosa.yin(self.audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_promedio = np.mean(f0[f0 > 0])  # Calcular f0 promedio ignorando valores cero
        print(f"Frecuencia fundamental detectada: {f0_promedio} Hz")
        return f0_promedio

    def normalizar_tono(self):
        # Extraer la frecuencia fundamental (f0), espectro, y aperiodicidad
        f0, sp, ap = pw.wav2world(self.audio.astype(np.float64), self.frecuencia_muestreo)

        # Calcular el factor de ajuste de f0
        f0_mean = np.mean(f0[f0 > 0])  # Media de f0 no nula
        f0_ratio = self.f0_deseada / f0_mean if f0_mean > 0 else 1

        # Ajustar f0 para que coincida con la f0 deseada
        f0_adjusted = f0 * f0_ratio

        # Resintetizar el audio con el pitch ajustado
        self.audio = pw.synthesize(f0_adjusted, sp, ap, self.frecuencia_muestreo)
        print("Pitch normalizado al valor deseado.")

    def reducir_ruido(self, ruido=None):
        if self.audio is not None and len(self.audio) > 0:
            if ruido is None:
                ruido = self.audio[:int(0.5 * self.frecuencia_muestreo)]
            ruido_medio = np.mean(ruido)
            self.audio = self.audio - ruido_medio
            print("Reducción de ruido aplicada.")

    def dividir_y_calcular_zcr(self, segmentos=5):
        if self.audio is None or len(self.audio) == 0:
            return [0] * segmentos

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

    def dividir_y_calcular_mfcc(self, segmentos=5, n_mfcc=3):
        if self.audio is None or len(self.audio) == 0:
            return np.zeros(segmentos * n_mfcc)

        duracion_segmento = int(self.frecuencia_muestreo * 0.1)  # 0.1 segundos por segmento
        mfcc_valores = []

        for i in range(segmentos):
            inicio = i * duracion_segmento
            fin = inicio + duracion_segmento
            segmento = self.audio[inicio:fin]
            
            if len(segmento) > 0:
                mfcc = librosa.feature.mfcc(y=segmento, n_fft = 1024, sr=self.frecuencia_muestreo, n_mfcc=n_mfcc)
                mfcc_valores.extend(np.mean(mfcc, axis=1))
            else:
                mfcc_valores.extend([0] * n_mfcc)

        print("MFCC calculado para cada segmento de 0.1 segundos.")
        return np.array(mfcc_valores)
    
    def guardar_audio_filtrado(self, ruta_guardado):
        if self.audio is not None and self.frecuencia_muestreo is not None:
            sf.write(ruta_guardado, self.audio, self.frecuencia_muestreo)
            print(f"Audio filtrado guardado en {ruta_guardado}")
        else:
            print("No hay audio filtrado para guardar.")

    def calcular_formantes(self, lpc_order=16):
        """Calcula los primeros tres formantes usando LPC"""
        # Calcular coeficientes LPC
        a = librosa.lpc(self.audio, order=lpc_order)
        # Calcular las raíces del polinomio
        roots = np.roots(a)
        # Filtrar raíces para obtener solo las frecuencias de formantes
        roots = [r for r in roots if np.imag(r) >= 0]
        angles = np.angle(roots)
        frequencies = sorted(angles * (self.frecuencia_muestreo / (2 * np.pi)))
        formantes = frequencies[:3] if len(frequencies) >= 3 else frequencies + [0] * (3 - len(frequencies))
        print("Formantes calculados:", formantes)
        return formantes

    def extraer_features(self):
        self.cargar_audio()
        self.recortar_por_volumen()
        self.reducir_ruido()
        self.normalizar_tono()
        self.aplicar_filtro()
       

        # Características segmentadas
        mfcc_features = self.dividir_y_calcular_mfcc(segmentos=5, n_mfcc=3)  # 15 características
        #
        # Características no segmentadas (formantes)
        formant_features = self.calcular_formantes()  # 3 características

        # Concatenación final
        self.caracteristicas = np.concatenate([mfcc_features, formant_features])
        print(f"Características extraídas: {self.caracteristicas}")
        return self.caracteristicas
import numpy as np
import librosa as lb
import os
import soundfile as sf



class Filtrado:
    
    def __init__(self, directorio_crudos, directorio_prefiltrados):
        self.directorio_crudos = '../anexos/test_raw'
        self.directorio_prefiltrados = '../anexos/test_filt'
        self.f0_promedio = None

    def filtrar_y_normalizar(self, audio, sr, f0_objetivo,filename):
        try:
            # Paso 1: Aplicar pre-énfasis
            audio_filtrado = lb.effects.preemphasis(audio, coef=0.4)
            
            # Paso 2: Normalizar el volumen
            peak_amplitude = np.max(np.abs(audio_filtrado))
            if peak_amplitude > 0:
                audio_filtrado = audio_filtrado / peak_amplitude

            # Paso 3: Calcular y ajustar el tono
            f0_actual = self.calcular_f0(audio_filtrado, sr,filename)


            if f0_actual and not np.isnan(f0_actual):
                factor = f0_objetivo / f0_actual
            else:
                print("Advertencia, no se pudo calcular f0 actual, usando f0_actual = 70")
                factor = f0_objetivo / 70.0  # Frecuencia promedio de la voz humana
            
            if 0.8 > factor > 1.5:  # Limitar a un rango razonable
                factor  = 1.0
                
            n_steps = np.log2(factor) * 12
            
            if -36 <= n_steps <= 36:  # Limitar a un rango razonable
                    audio_filtrado = lb.effects.pitch_shift(audio_filtrado, sr=sr, n_steps=n_steps)
            
            else:
                    print(f"Advertencia: Cambio de tono fuera de rango ({n_steps} semitonos). No se aplicará.")
            
            return audio_filtrado  # Ensure the function returns the processed audio

        except Exception as e:
            print(f"Error al filtrar y normalizar audio: {e}")
            return None

    def filtrar_audio(self,audio,sr):
        try:
            audio_filtrado = lb.effects.preemphasis(audio, coef=0.4)
            return audio_filtrado
        except Exception as e:
            print(f"Error al filtrar audio: {e}")
            return None

    def calcular_f0(self, audio, sr,filename):
        try:
            f0, _, _ = lb.pyin(audio, fmin=lb.note_to_hz('C2'), fmax=lb.note_to_hz('C7'))
            print(f"{filename}: f0 calculada -> {np.nanmean(f0)}")
            return np.nanmean(f0) if f0 is not None else None
        except Exception as e:
            print(f"Error al calcular f0: {e}")
            return None

    def procesar_audios(self):
        f0_totales = []
        os.makedirs(self.directorio_prefiltrados, exist_ok=True)

        # Paso 1: Calcular la f0 promedio de los audios crudos
        try:      
            for root, dirs, files in os.walk(self.directorio_crudos):
                for file in files:
                    if file.endswith(".wav"):
                        ruta = os.path.join(root, file)
                        audio, sr = lb.load(ruta, sr=None)

                        audio_filtrado = self.filtrar_audio(audio, sr)
                        # Calcular f0 del audio crudo
                        filename = file
                        f0 = self.calcular_f0(audio_filtrado, sr,filename)
                        if f0 is not None and not np.isnan(f0):
                            f0_totales.append(f0)

            self.f0_promedio = np.mean(f0_totales) if f0_totales else None
            print(f"f0 promedio calculada: {self.f0_promedio}")
        
        except Exception as e:
            print(f"Error al calcular f0 promedio: {e}")
            return None
        
        # Paso 2: Filtrar y normalizar cada audio crudo
       
        try:
            for root, dirs, files in os.walk(self.directorio_crudos):
                for file in files:
                    if file.endswith(".wav"):
                        ruta = os.path.join(root, file)
                        audio, sr = lb.load(ruta, sr=None)

                        # Filtrar y normalizar el audio
                        audio_normalizado = self.filtrar_y_normalizar(audio, sr, self.f0_promedio,file)
                        if audio_normalizado is None:
                            print(f"Advertencia: No se pudo procesar el audio {file}. Será ignorado.")
                            continue
                        
                        # Guardar el audio preprocesado (filtrado y normalizado)
                        ruta_guardado = os.path.join(self.directorio_prefiltrados, file)
                        sf.write(ruta_guardado, audio_normalizado, sr)
                        print(f"Guardado audio filtrado y normalizado en: {ruta_guardado}")
                    
        except Exception as e:  
            print(f"Error al procesar audios: {e}")
            return None
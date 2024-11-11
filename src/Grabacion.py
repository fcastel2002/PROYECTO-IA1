import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time  # Add this import to calculate duration

class Grabacion:
    def __init__(self, frecuencia_muestreo=44100):
        self.frecuencia_muestreo = frecuencia_muestreo
        self.audio = None
        self.start_time = None

    def grabar(self):
        print("Iniciando grabación...")
        self.start_time = time.time()
        self.audio = sd.rec(int(10 * self.frecuencia_muestreo), samplerate=self.frecuencia_muestreo, channels=1, dtype='float64')  # Start recording with an arbitrary long duration

    def detener(self):
        sd.stop()
        duracion = time.time() - self.start_time
        self.audio = self.audio[:int(duracion * self.frecuencia_muestreo)]  # Trim the audio to the actual duration
        print("Grabación finalizada.")

    def guardar(self, nombre_archivo):
        escalado = np.int16(self.audio / np.max(np.abs(self.audio)) * 32767)
        write('../anexos/audios/'+nombre_archivo+'.wav', self.frecuencia_muestreo, escalado)
        print(f"Audio guardado como {nombre_archivo}.")

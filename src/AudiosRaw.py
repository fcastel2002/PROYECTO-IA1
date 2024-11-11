import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import os
import keyboard  # Add this import to detect key presses

class AudiosRaw:
    def __init__(self, frecuencia_muestreo=44100):
        self.frecuencia_muestreo = frecuencia_muestreo
        self.audio = None
        self.start_time = None
        self.counter = 0
        self.folder_path = '../anexos/audios_raw/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def grabar(self):
        print("Iniciando grabación...")
        self.start_time = time.time()
        self.audio = sd.rec(int(10 * self.frecuencia_muestreo), samplerate=self.frecuencia_muestreo, channels=1, dtype='float64')

    def detener(self):
        sd.stop()
        duracion = time.time() - self.start_time
        self.audio = self.audio[:int(duracion * self.frecuencia_muestreo)]
        print("Grabación finalizada.")

    def guardar(self):
        nombre_archivo = f"berenjena_{self.counter}.wav"
        escalado = np.int16(self.audio / np.max(np.abs(self.audio)) * 32767)
        write(os.path.join(self.folder_path, nombre_archivo), self.frecuencia_muestreo, escalado)
        print(f"Audio guardado como {nombre_archivo}.")
        self.counter += 1

    def iniciar(self):
        print("Mantén presionada la tecla ESPACIO para grabar. Suelta la tecla para detener la grabación. Presiona ESC para salir.")
        while True:
            if keyboard.is_pressed('space'):
                self.grabar()
                while keyboard.is_pressed('space'):
                    time.sleep(0.1)
                self.detener()
                self.guardar()
            elif keyboard.is_pressed('esc'):
                print("Saliendo...")
                break

# Example usage
if __name__ == "__main__":
    grabador = AudiosRaw()
    grabador.iniciar()

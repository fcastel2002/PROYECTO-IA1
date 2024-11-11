import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import keyboard

# Configuración de grabación
fs = 44100  # Frecuencia de muestreo (Hz)
audio_data = []  # Lista para almacenar los datos de audio grabados
is_recording = False  # Estado de grabación

def start_recording():
    global is_recording, audio_data
    print("Grabación iniciada...")
    is_recording = True
    audio_data = []  # Limpiar datos anteriores

def stop_recording():
    global is_recording
    print("Grabación detenida.")
    is_recording = False
    # Convertir audio_data a un array numpy y guardarlo en un archivo .wav
    audio_np = np.concatenate(audio_data, axis=0)
    write("grabacion.wav", fs, audio_np)
    print("Audio guardado como 'grabacion.wav'")

def callback(indata, frames, time, status):
    """Callback de grabación que almacena los datos de audio mientras se graba"""
    if is_recording:
        audio_data.append(indata.copy())

# Configuración de la función de grabación con sounddevice
stream = sd.InputStream(samplerate=fs, channels=1, callback=callback)

print("Presiona la barra espaciadora para iniciar y detener la grabación.")
with stream:
    while True:
        if keyboard.is_pressed("space"):
            if not is_recording:
                start_recording()
            else:
                stop_recording()
            # Espera a que se suelte la barra espaciadora para evitar múltiples activaciones
            while keyboard.is_pressed("space"):
                pass

# Comparacion de procesamiento secuencial y paralelo

import time
import multiprocessing as mp
import pandas as pd

from src.preprocesamiento import preprocesar

# Procesar una imagen sin devolver resultados
def procesar_sin_resultado(ruta):
    preprocesar(ruta)

if __name__ == "__main__":

    # Leer dataset
    datos = pd.read_csv("data/dataset.csv")

    # Cantidades de imagenes
    cantidades = [1000, 3000, 5000]

    for cantidad in cantidades:

        rutas = datos["ruta"].head(cantidad).tolist()

        # Procesamiento secuencial
        inicio = time.perf_counter()

        for ruta in rutas:
            preprocesar(ruta)

        tiempo_secuencial = time.perf_counter() - inicio

        # Procesamiento paralelo
        inicio = time.perf_counter()

        with mp.Pool(processes=4) as pool:
            for resultado in pool.imap_unordered(procesar_sin_resultado, rutas):
                pass

        tiempo_paralelo = time.perf_counter() - inicio

        print("Imagenes:", cantidad)
        print("Tiempo secuencial:", round(tiempo_secuencial, 4), "segundos")
        print("Tiempo paralelo:", round(tiempo_paralelo, 4), "segundos")
        
        # Calcular la aceleracion
        speedup = tiempo_secuencial / tiempo_paralelo

        # Calcular la eficiencia
        eficiencia = (speedup / 4) * 100

        print("Speedup:", round(speedup, 2))
        print("Eficiencia:", round(eficiencia, 2), "%")
        print("---------------------------")
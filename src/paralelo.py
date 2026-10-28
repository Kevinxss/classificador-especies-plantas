import os
import time 
import multiprocessing as mp
import pandas as pd
import numpy as np

from src.caracteristicas import procesar_imagen
from src.config import RAIZ

Procesos = 4
n_imagenes = None
dataset_cvs = RAIZ / "data" / "dataset.csv"
salida = RAIZ / "data" / "caracteristicas.npy"
comparar = True

def extraer_secuencial(ruta):
    vectores = [procesar_imagen(ruta) for ruta in ruta]
    return np.vstack(vectores)

def extraer_paralelo(rutas, procesos=Procesos):
    with mp.Pool(processes=procesos) as pool:
        resultados = pool.map(procesar_imagen, rutas)
    return np.vstack(resultados)


if __name__ == "__main__":
    datos = pd.read_csv(dataset_cvs)
    if n_imagenes:
        datos = datos.sample(n=n_imagenes, random_state=42)
    rutas = datos["ruta"].tolist()
    print("Imagenes- ", {len(rutas)}, "-Procesos- ", {Procesos}, "- Salida- ", {salida})


    inicio = time.perf_counter()
    x = extraer_paralelo(rutas, procesos=Procesos)
    tiempo_paralelo = time.perf_counter() - inicio
    print("Tiempo paralelo:", round(tiempo_paralelo, 4), "segundos")

    if n_imagenes is None:
        np.save(salida, x)
        print("Guardado en", salida)

    if comparar:
        inicio = time.perf_counter()
        X_sec = extraer_secuencial(rutas)
        tiempo_secuencial = time.perf_counter() - inicio
 
        speedup = tiempo_secuencial / tiempo_paralelo
        eficiencia = speedup / Procesos * 100
        print("Tiempo secuencial:", round(tiempo_secuencial, 2), "s")
        print("Speedup:", round(speedup, 2), "| Eficiencia:", round(eficiencia, 1), "%")
        print("Mismo resultado que secuencial:", np.array_equal(x, X_sec))





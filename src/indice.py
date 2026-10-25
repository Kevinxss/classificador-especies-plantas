from pathlib import Path #recorrer carpetas 
import csv

from src.config import DATA_DIR #dataset 

#organizar la inf del datset 
def construir_indice():
    datos = []  #list para inf de imgs

    for carpeta in DATA_DIR.iterdir(): #recorremos las carpetas 
        if carpeta.is_dir():
            clase = carpeta.name #nombre de la carpeta 
            especie = clase.split("___")[0] #tipo de especie 

            if "healthy" in clase.lower(): #sana o enferma 
                estado = "sano"
            else:
                estado = "enfermo"

            for imagen in carpeta.iterdir():
                if imagen.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    ruta = imagen.relative_to(DATA_DIR)
                    datos.append([str(ruta), especie, estado, clase])


        archivo = DATA_DIR.parent / "dataset.csv"

    with open(archivo, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f) #escritura con formato csv 
        escritor.writerow(["ruta", "especie", "estado", "clase"])
        escritor.writerows(datos)

    print("Imagenes registradas:", len(datos))

if __name__ == "__main__":
    construir_indice()
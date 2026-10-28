import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from src.config import RAIZ

semilla = 42
n_arboles = 100
prueba = 0.2
balanceado = True
procesadores = 4
pliegues = 3

CARACTERISTICAS_NPY = RAIZ / "data" / "caracteristicas.npy"
DATASET_CSV = RAIZ / "data" / "dataset.csv"
METRICAS_DIR = RAIZ / "resultados" / "metricas"
FIGURAS_DIR = RAIZ / "resultados" / "figuras"


def nuevo_modelo():
    return RandomForestClassifier(n_estimators=n_arboles, random_state=semilla, class_weight="balanced" if balanceado else None)

def evaluar(objetivo, x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, 
        test_size=prueba,
        random_state=semilla,
        stratify=y)


    modelo = nuevo_modelo()
    inicio = time.perf_counter()
    modelo.fit(x_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - inicio

    y_pred = modelo.predict(x_test)

    metricas = {
            "tiempo_entrenamiento": tiempo_entrenamiento,
            "exactitud": modelo.score(x_test, y_test),
            "matriz_confusion": confusion_matrix(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted")

        }
    with open(METRICAS_DIR / f"reporte_{objetivo}.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, digits=3, zero_division=0))
 
    graficar_confusion(objetivo, y_test, y_pred, modelo.classes_)
    return metricas

    
def graficar_confusion(objetivo, y_real, y_pred, clases):
    # normalizada por fila: cada fila muestra a qué se fueron las imágenes de esa clase real
    cm = confusion_matrix(y_real, y_pred, labels=clases, normalize="true")
    n = len(clases)
    lado = max(4.5, 0.55 * n + 2)
    fig, ax = plt.subplots(figsize=(lado, lado * 0.9))
    im = ax.imshow(cm, vmin=0, vmax=1)
    ax.set_xticks(range(n), clases, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n), clases, fontsize=9)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión — {objetivo} (normalizada por fila)",  fontsize=11, loc="left")
    for i in range(n):
        for j in range(n):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8 if n < 10 else 6,
                        color="white" if cm[i, j] > 0.55 else "black" )
    for borde in ax.spines.values():
        borde.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"confusion_{objetivo}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    METRICAS_DIR.mkdir(parents=True, exist_ok=True)   
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    x = np.load(CARACTERISTICAS_NPY)
    datos = pd.read_csv(DATASET_CSV)
    print("Matriz de características:", x.shape)

    for objetivo in ["especie", "estado"]:
        y = datos[objetivo].to_numpy()
        metricas = evaluar(objetivo, x, y)
        print(f"\n== {objetivo} ==")
        print("Exactitud:", round(metricas["exactitud"], 3))
        print("F1:", round(metricas["f1_score"], 3))
        print("Tiempo de entrenamiento:", round(metricas["tiempo_entrenamiento"], 1), "s")
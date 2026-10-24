from pathlib import Path

import cv2
import numpy as np
from src.config import DATA_DIR, TAMANO_IMG


cv2.setNumThreads(1)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def leer_imagen(ruta):
    #leemos la imagen bgr
    ruta = Path(ruta)
    if not ruta.is_absolute():
        ruta = DATA_DIR/ruta
        img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if img is None:
        raise FileExistsError("No se encontro la imagen")
    return img



def mascara_hoja(img):
    """Máscara binaria (0 = fondo, 255 = hoja) con Otsu sobre el exceso de verde."""
    b, g, r = cv2.split(img.astype(np.int16))
    exg = np.clip(2 * g - r - b, 0, 255).astype(np.uint8)
    _, mascara = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, KERNEL)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, KERNEL)
    return mascara
 
 
def preprocesar_imagen(img):
    """Regresa (hsv, gris, mascara) a TAMANO_IMG, todos uint8."""
    img = cv2.resize(img, TAMANO_IMG, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mascara = mascara_hoja(img)
    return hsv, gris, mascara
 
 
def preprocesar(ruta):
    """Lee la imagen de `ruta` y la preprocesa."""
    return preprocesar_imagen(leer_imagen(ruta))


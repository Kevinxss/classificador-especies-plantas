import numpy as np
import cv2
from skimage.feature import local_binary_pattern

from.preprocesamiento import preprocesar

b_h, b_s, b_v = 16, 8 , 8
lbp_p, lbp_r = 8,1
n_caracteristicas = b_h + b_s + b_v + (lbp_p + 2) + 5

def normalizar(hist):
    total = hist.sum()
    return hist / total if total > 0 else hist

def color_hsv(hsv, mascara):
    hist_h = cv2.calcHist([hsv], [0], mascara, [b_h], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], mascara, [b_s], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], mascara, [b_v], [0, 256]).flatten()

    hist_h = normalizar(hist_h)
    hist_s = normalizar(hist_s)
    hist_v = normalizar(hist_v)

    return np.concatenate((hist_h.flatten(), hist_s.flatten(), hist_v.flatten()))   

def textura_lbp(gris, mascara):
    lbp = local_binary_pattern(gris, lbp_p, lbp_r, method="uniform")
    hist_lbp, _ = np.histogram(lbp[mascara > 0], bins=lbp_p + 2, range=(0, lbp_p + 2))
    hist_lbp = normalizar(hist_lbp.astype(np.float32))
    return hist_lbp
def forma(mascara):
    momentos = cv2.moments(mascara)
    hu_moments = cv2.HuMoments(momentos).flatten()
    return hu_moments

def procesar_imagen(ruta):
    hsv, gris, mascara = preprocesar(ruta)

    caracteristicas_color = color_hsv(hsv, mascara)
    caracteristicas_textura = textura_lbp(gris, mascara)
    caracteristicas_forma = forma(mascara)

    caracteristicas = np.concatenate([caracteristicas_color,
                                      caracteristicas_textura,
                                      caracteristicas_forma])
    return caracteristicas.astype(np.float32)


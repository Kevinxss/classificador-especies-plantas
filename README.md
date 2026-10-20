# Clasificación de especies de plantas con aprendizaje automático tradicional

> **Estado del proyecto:** propuesta (fase de planeación). Aún no hay código ni resultados.

**Materia:** Computo Paralelo \n
**Institución:** Unidad profesional interdisiplinaria de ingenerias compus Tlaxcala
**Profesor:** Jesus Garcia \n
**Equipo:** \n Equipo classificador-especies-plantas


---

## Descripción

Este proyecto propone un sistema que clasifica **14 especies de plantas** a partir de imágenes de hojas del conjunto de datos **PlantVillage**, utilizando únicamente **aprendizaje automático tradicional** (sin deep learning). Además, se **paralelizan** las etapas más costosas del proceso para reducir el tiempo de cómputo y se mide la mejora obtenida.

## Objetivos

**General:** desarrollar y evaluar un clasificador de especies de plantas basado en características extraídas manualmente de imágenes de hojas.

**Específicos:**
- Extraer un conjunto de características simples de color, textura y forma.
- Usar solo aprendizaje maquina como lo es el Random Forest
- Paralelizar la extracción de características y la búsqueda de hiperparámetros.
- Evaluar el desempeño de los modelos y el beneficio de la paralelización (speedup y eficiencia).

## Conjunto de datos

[PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset): imágenes RGB de hojas de 256×256 píxeles, etiquetadas en 38 clases (especie y estado de la hoja). Para este proyecto se agrupan por especie, quedando **14 clases**:

> El conjunto de datos **no se incluye en este repositorio**; debe descargarse desde su fuente original.

## Metodología



| Etapa | Descripción |
|---|---|
| **Preprocesamiento** | Redimensión de las imágenes, conversión a HSV y escala de grises, y obtención de la máscara de la hoja. |
| **Extracción de características (51)** | Color: histogramas de H, S y V (32). Textura: histograma LBP uniforme (10). Forma: momentos de Hu, área relativa y relación de aspecto (9). |
| **Partición** | División estratificada: 80 % entrenamiento y 20 % prueba. |
| **Modelos** | SVM con kernel RBF (con estandarización) y Random Forest. |
| **Validación** | Validación cruzada estratificada de 5 pliegues con búsqueda en malla (`GridSearchCV`), usando F1 macro como criterio. |
| **Evaluación** | Exactitud, precisión, exhaustividad, F1 macro y matriz de confusión sobre el conjunto de prueba. |
| **Paralelización** | Extracción de características con `joblib`, y `n_jobs` en la búsqueda de hiperparámetros y en el Random Forest. Se comparará la ejecución secuencial contra la paralela. |

**Métricas de paralelización:**

- Aceleración (speedup): `S(p) = T₁ / T_p`
- Eficiencia: `E(p) = S(p) / p`

donde `T₁` es el tiempo secuencial y `T_p` el tiempo con `p` procesos.

## Tecnologías

Python · scikit-learn · OpenCV · scikit-image · NumPy · joblib · Matplotlib




## Equipo

| Integrante | Responsabilidad |
|---|---|
| [Kevin Velazquez Gonzalez] 
| [Josue David Rodriguez Vezquez] 


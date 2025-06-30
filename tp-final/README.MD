# Trabajo Práctico Final – Matemática Aplicada y Optimización

Este repositorio contiene el desarrollo completo del trabajo práctico final de la materia **Matemática Aplicada y Optimización**, correspondiente a la Maestría en Inteligencia Artificial (Universidad de San Andrés).

El objetivo del TP es estudiar el comportamiento del entrenamiento de una red neuronal simple (MLP) sobre una tarea de reconstrucción de imagen, comparando el rendimiento de diferentes optimizadores (SGD, RMSprop y Adam), e incorporando análisis adicionales como visualización de la función de costo, distribución de activaciones y exploración de hiperparámetros.

El entregable principal del trabajo es el archivo PDF (`Trabajo Práctico Final - Matemática Aplicada y Optimización.pdf`), donde se detallan los experimentos, resultados, análisis y conclusiones. Este repositorio acompaña ese documento con todo el código reproducible.

---

## Estructura de archivos

### `experiments.py`
Archivo principal para ejecutar los distintos experimentos. Define un listado de configuraciones (`experiments`) con distintos hiperparámetros (optimizer, learning rate, batch size, etc.), y llama a la función `run_experiment(config)` para correr cada uno. Es el punto de entrada para probar variantes del entrenamiento de la red.

### `simple_nn_2d.py`
Contiene la función `run_experiment(config)`, que realiza el entrenamiento completo de la red neuronal según los parámetros definidos. Incluye:
- Preparación de los datos (`field.npy`)
- Inicialización de pesos
- Entrenamiento por épocas con minibatches
- Evaluación del loss
- Visualización de:
  - Evolución de la función de pérdida
  - Comparación entre imagen original y predicha
  - Histograma de activaciones neuronales
  - Superficie local de la función de costo en 3D

Todos los gráficos se guardan automáticamente en la carpeta `~/Desktop/results/`.

### `nn_functions.py`
Define las funciones auxiliares utilizadas por la red:
- Inicialización de parámetros
- Forward pass (`predict`)
- Cálculo del `loss` (MSE)
- Implementaciones de `SGD`, `RMSprop`, `Adam`
- Batching y manejo de parámetros como vectores
- Función para obtener activaciones de capas ocultas

### `field.npy`
Archivo `.npy` con el campo escalar objetivo (la imagen a reconstruir). Representa una grilla 2D de valores reales. Se utiliza como "ground truth" para entrenar y evaluar la red.

### `Enunciado.pdf`
Archivo original con la consigna oficial del trabajo práctico.

### `Trabajo Práctico Final - Matemática Aplicada y Optimización.pdf`
Entregable final del trabajo, con desarrollo teórico, implementación, gráficos, análisis comparativo de optimizadores, y conclusiones.

---

## ¿Cómo correr un experimento?

1. Asegurarse de tener instalado `jax`, `matplotlib` y `numpy`.
2. Editar `experiments.py` con los experimentos que se desean correr.
3. Ejecutar:

```bash
python experiments.py
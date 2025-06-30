# simple_nn_2d.py

# Importo las librerías necesarias para cálculo numérico, derivación automática y visualización
import jax.numpy as jnp
from jax import grad, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # para gráfico 3D de la loss function

# Importo desde nn_functions.py todas las funciones y estructuras necesarias para definir y entrenar la red
from nn_functions import (
    init_network_params, pack_params, layer_sizes,
    update_sgd, update_rmsprop, update_adam,
    get_batches, loss, batched_predict, get_all_activations
)

def run_experiment(config):
    """
    Esta función ejecuta un experimento de entrenamiento de red neuronal sobre un campo escalar.
    Utiliza los hiperparámetros especificados en el diccionario `config`.
    """

    # ---------- CARGA Y NORMALIZACIÓN DE DATOS ----------

    # Cargo el campo escalar desde archivo .npy. Asumo que representa una función sobre una grilla 2D.
    field = jnp.load('field.npy')

    # Centralizo el campo restando la media y luego lo escalo dividiendo por su desvío estándar
    # Esto mejora la estabilidad numérica del entrenamiento
    field = field - field.mean()
    field = field / field.std()
    field = jnp.array(field, dtype = jnp.float32)

    # Obtengo las dimensiones del campo
    nx, ny = field.shape

    # Construyo una malla de coordenadas normalizadas entre -1 y 1 en ambas direcciones
    xx = jnp.linspace(-1, 1, nx)
    yy = jnp.linspace(-1, 1, ny)
    xx, yy = jnp.meshgrid(xx, yy, indexing = 'ij')

    # Concateno las coordenadas x e y para formar los inputs a la red: cada punto es un vector (x, y)
    xx = jnp.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis = 1)

    # Aplano el campo original para que coincida con el formato esperado por la red (vector columna)
    ff = field.reshape(-1, 1)

    # ---------- INICIALIZACIÓN DE PARÁMETROS Y OPTIMIZADOR ----------

    # Inicializo los parámetros de la red con el esquema de Glorot usando una semilla fija
    key = random.PRNGKey(config["seed"])
    params = init_network_params(layer_sizes, key)

    # Convierto los parámetros a un vector plano, necesario para mis funciones de optimización
    params = pack_params(params)

    # Selecciono el optimizador a partir del nombre pasado en la configuración
    update = {
        "sgd": update_sgd,
        "rmsprop": update_rmsprop,
        "adam": update_adam
    }[config["optimizer"]]

    # ---------- INICIALIZACIÓN DE VARIABLES AUXILIARES PARA EL OPTIMIZADOR ----------

    # Tomo un primer mini-batch para calcular los gradientes iniciales
    xi, yi = next(get_batches(xx, ff, bs = config["batch_size"]))

    # Calculo los gradientes del loss respecto a los parámetros
    grads = grad(loss)(params, xi, yi)

    # Inicializo la variable auxiliar 'aux' según el optimizador
    if config["optimizer"] == "adam":
        aux = (0, jnp.zeros_like(grads), jnp.zeros_like(grads))  # t, m, v
    elif config["optimizer"] == "rmsprop":
        aux = jnp.square(grads)
    else:
        aux = grads * 0  # aux no cambia en SGD, pero lo dejo por consistencia

    # ---------- ENTRENAMIENTO ----------

    # Creo una lista para guardar la pérdida de entrenamiento en cada época
    log_train = []

    # Itero sobre las épocas
    for epoch in range(config["num_epochs"]):
        # Para cada época, mezclo aleatoriamente los índices del dataset (permuto las filas)
        key, subkey = random.split(key)
        idxs = random.permutation(subkey, xx.shape[0])

        # Itero sobre mini-batches generados con los datos permutados
        for xi, yi in get_batches(xx[idxs], ff[idxs], bs = config["batch_size"]):
            # Actualizo los parámetros usando el optimizador seleccionado
            if config["optimizer"] == "adam":
                params, aux = update(
                    params, xi, yi, config["lr"], aux,
                    config.get("beta1", 0.9),
                    config.get("beta2", 0.999)
                )
            else:
                params, aux = update(params, xi, yi, config["lr"], aux)

        # Calculo la pérdida total sobre todos los datos al final de cada época
        train_loss = loss(params, xx, ff)
        log_train.append(train_loss)

        # Imprimo la pérdida actual para tener seguimiento del progreso del entrenamiento
        print(f"[{config['name']}] Epoch {epoch}, Loss: {train_loss:.6f}")

    # ---------- RESUMEN FINAL ----------

    # Imprimo un resumen con los hiperparámetros utilizados y la pérdida final alcanzada
    print("\n RESUMEN DEL EXPERIMENTO:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  Final Loss: {log_train[-1]:.6f}\n")

    # ---------- VISUALIZACIÓN DE RESULTADOS ----------

    import os

    # Defino el directorio donde se guardarán los gráficos (Desktop/results)
    results_dir = os.path.expanduser("~/Desktop/results")
    os.makedirs(results_dir, exist_ok = True)

    # Aseguro que el nombre del archivo sea seguro para el sistema de archivos
    safe_name = config["name"].replace(" ", "_").replace("/", "_").replace("=", "")

    # LOSS FUNCTION
    # Genero y guardo la evolución de la pérdida (en escala logarítmica)
    fig = plt.figure(figsize = (6, 4))
    plt.plot(log_train, linewidth=1.2)
    plt.title("Training Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{safe_name}_Loss.png"))

    # ORIGINAL VS PREDICTED
    # Genero y guardo una figura comparativa: campo real vs predicción
    pred = batched_predict(params, xx).reshape((nx, ny)).T
    true = ff.reshape((nx, ny)).T

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(true, origin='lower', cmap='jet')
    axes[0].set_title("Original Field")
    axes[0].axis('off')

    axes[1].imshow(pred, origin='lower', cmap='jet')
    axes[1].set_title("Neural Network Prediction")
    axes[1].axis('off')

    plt.suptitle(config["name"])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{safe_name}_Comparison.png"))

    # ACTIVATIONS
    # Obtengo todas las activaciones ocultas luego del entrenamiento
    all_activations = get_all_activations(params, xx)

    # Grafico el histograma de valores de activación
    plt.figure(figsize=(8, 4))
    plt.hist(all_activations.ravel(), bins=100, color='skyblue', edgecolor='black')
    plt.title("Neural Activation Distribution\n" + config["name"])
    plt.xlabel("Activation value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{safe_name}_Activations.png"))

    # LOSS FUNCTION SURFACE
    # Defino dos direcciones aleatorias en el espacio de parámetros (misma dimensión)
    d1_key, d2_key = random.split(key)
    d1 = random.normal(d1_key, shape=params.shape)
    d2 = random.normal(d2_key, shape=params.shape)

    # Normalizo las direcciones para que sean comparables
    d1 = d1 / jnp.linalg.norm(d1)
    d2 = d2 / jnp.linalg.norm(d2)

    # Defino la grilla de alfas y betas (de -1 a 1)
    n = 30
    alphas = jnp.linspace(-3.0, 3.0, n)
    betas = jnp.linspace(-3.0, 3.0, n)
    A, B = jnp.meshgrid(alphas, betas, indexing="ij")

    # Calculo la loss en cada combinación de dirección (alfa * d1 + beta * d2)
    losses = jnp.zeros((n, n))
    for i in range(n):
        for j in range(n):
            perturbed_params = params + alphas[i] * d1 + betas[j] * d2
            losses = losses.at[i, j].set(loss(perturbed_params, xx, ff))

    # Grafico la superficie
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, losses, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_title("Loss Surface")
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_zlabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{safe_name}_LossSurface.png"))
    plt.close(fig)

    # Cierro las figuras para liberar memoria y evitar que se abran
    plt.close('all')
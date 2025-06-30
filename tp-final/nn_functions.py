# nn_functions.py

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn

# ---------- ESTRUCTURA DE LA RED ----------

layer_sizes = [2, 64, 64, 1]

def pack_params(params):
    return jnp.concatenate([jnp.ravel(w) for w, _ in params] +
                           [jnp.ravel(b) for _, b in params])

def unpack_params(params):
    weights = []
    for i in range(len(layer_sizes) - 1):
        weight_size = layer_sizes[i] * layer_sizes[i + 1]
        to_unpack, params = params[:weight_size], params[weight_size:]
        weights.append(jnp.array(to_unpack).reshape(layer_sizes[i + 1], layer_sizes[i]))

    biases = []
    for i in range(len(layer_sizes) - 1):
        bias_size = layer_sizes[i + 1]
        to_unpack, params = params[:bias_size], params[bias_size:]
        biases.append(jnp.array(to_unpack).reshape(layer_sizes[i + 1]))

    return [(w, b) for w, b in zip(weights, biases)]

def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    scale = jnp.sqrt(6.0 / (m + n))
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# ---------- FORWARD ----------

@jit
def predict(params, coord):
    params = unpack_params(params)
    hidden = coord
    for w, b in params[:-1]:
        hidden = nn.tanh(jnp.dot(w, hidden) + b)
    final_w, final_b = params[-1]
    return jnp.dot(final_w, hidden) + final_b

batched_predict = vmap(predict, in_axes=(None, 0))

# ---------- LOSS ----------

def loss(params, coord, target):
    preds = batched_predict(params, coord)
    return jnp.mean(jnp.square(preds - target))

# ---------- OPTIMIZADORES ----------

@jit
def update_sgd(params, x, y, step_size, aux):
    grads = grad(loss)(params, x, y)
    return params - step_size * grads, aux

@jit
def update_rmsprop(params, x, y, step_size, aux):
    beta = 0.9
    grads = grad(loss)(params, x, y)
    aux = beta * aux + (1 - beta) * jnp.square(grads)
    step = step_size / (jnp.sqrt(aux) + 1e-8)
    return params - step * grads, aux

@jit
def update_adam(params, x, y, step_size, aux, beta1 = 0.9, beta2 = 0.999):
    t, m, v = aux
    grads = grad(loss)(params, x, y)
    t += 1
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * jnp.square(grads)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    step = step_size * m_hat / (jnp.sqrt(v_hat) + 1e-8)
    return params - step, (t, m, v)

# ---------- BATCHING ----------

def get_batches(x, y, bs):
    for i in range(0, len(x), bs):
        yield x[i:i+bs], y[i:i+bs]

# -------- ACTIVATIONS ---------
def get_all_activations(params, inputs):
    """
    Recorro la red capa por capa y acumulo todas las activaciones ocultas
    (es decir, los outputs intermedios después de aplicar tanh).
    """
    activations = []
    params = unpack_params(params)
    hidden = inputs

    for w, b in params[:-1]:
        outputs = jnp.dot(hidden, w.T) + b
        hidden = nn.tanh(outputs)
        activations.append(hidden)

    # Concateno todas las activaciones de todas las capas ocultas
    return jnp.concatenate(activations, axis=1)
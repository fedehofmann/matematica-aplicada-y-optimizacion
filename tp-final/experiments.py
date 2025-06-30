# experiments.py

from simple_nn_2d import run_experiment

# Aquí va cada uno de los diccionarios que a su vez representan los distintos experimentos que quiero hacer
experiments = [
{
    "name": "SGD LR=0.001 / BS=64",
    "optimizer": "adam",
    "lr": 0.001,
    "batch_size": 64,
    "num_epochs": 500,
    "seed": 123,
    "beta1": 0.9,
    "beta2": 0.99
}
]

# Corro cada experimento definido
for config in experiments:
    run_experiment(config)
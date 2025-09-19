from optimizer.adam import create_optimizer as create_optimizer_adam
from optimizer.sgd import create_optimizer as create_optimizer_sgd


def get_optimizer_scheduler(name, params, learning_rate, epochs):
    if name == 'Adam':
        opt = create_optimizer_adam(params=params, learning_rate=learning_rate, epochs=epochs)
    elif name == 'SGD':
        opt = create_optimizer_sgd(params=params, learning_rate=learning_rate, epochs=epochs)
    else:
        raise ValueError(f"Optimizer {name} not supported.")
    return opt

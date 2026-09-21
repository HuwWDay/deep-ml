import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999,
                   epsilon=1e-8, num_iterations=10):
    x = x0
    mt = 0
    vt = 0

    for t in range(1, num_iterations + 1):
        gt = grad(x)
        mt = beta1 * mt + (1 - beta1) * gt
        vt = beta2 * vt + (1 - beta2) * (gt ** 2)

        hatmt = mt / (1 - beta1 ** t)
        hatvt = vt / (1 - beta2 ** t)

        x = x - learning_rate * hatmt / (np.sqrt(hatvt) + epsilon)

    return x

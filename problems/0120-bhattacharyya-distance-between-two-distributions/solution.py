import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if len(p) != len(q):
        return 0.0
    elif len(p) == 0 or len(q) == 0:
        return 0.0
    else:
        BC = sum([np.sqrt(a*b) for (a, b) in zip(p, q)])
        BD = -np.log(BC)
        return BD
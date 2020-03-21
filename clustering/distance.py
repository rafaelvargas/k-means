import numpy as np

distance_calculation_methods = {
    "euclidian": lambda p1, p2: np.sqrt(np.sum(np.power(p1 - p2, 2))),
    "manhattan": lambda p1, p2: np.sum(np.absolute(p1 - p2)),
}

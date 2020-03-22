import numpy as np

EARTH_MEAN_RADIUS = 6371.0088


def haversine(p1, p2, r=EARTH_MEAN_RADIUS):
    p1_radians = np.radians(p1)
    p2_radians = np.radians(p2)
    angle = 2 * np.arcsin(
        np.sqrt(
            np.power(np.sin(p2_radians[0] - p1_radians[0]), 2)
            + (
                np.cos(p1_radians[0])
                * np.cos(p2_radians[0])
                * np.power(np.sin(p2_radians[1] - p1_radians[1]), 2)
            )
        )
    )
    return angle * EARTH_MEAN_RADIUS


distance_calculation_methods = {
    "euclidian": lambda p1, p2: np.sqrt(np.sum(np.power(p1 - p2, 2))),
    "manhattan": lambda p1, p2: np.sum(np.absolute(p1 - p2)),
    "haversine": haversine,
}

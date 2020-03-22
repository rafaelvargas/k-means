import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from clustering.KmeansClusterer import KmeansClusterer

NUMBER_OF_POINTS = 1000

def generate_random_points(number_of_points, dimension):
    return np.random.rand(number_of_points, dimension)


if __name__ == "__main__":
    points = generate_random_points(NUMBER_OF_POINTS, 2)
    clusterer = KmeansClusterer(5, points, distance="haversine")
    clusters = clusterer.run(10)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for cluster in clusters:
        ax.scatter(cluster[0], cluster[1])

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    plt.title("2D K-means Clustering")
    plt.show()

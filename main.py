import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from clustering.KmeansClusterer import KmeansClusterer
from scipy.spatial import Voronoi, voronoi_plot_2d

NUMBER_OF_POINTS = 100


def generate_random_points(number_of_points, dimension):
    return np.random.rand(number_of_points, dimension)


if __name__ == "__main__":
    points = generate_random_points(NUMBER_OF_POINTS, 2)
    clusterer = KmeansClusterer(
        5, points, distance="euclidian", weights=np.random.rand(NUMBER_OF_POINTS)
    )
    clusters, clusters_weights, centroids = clusterer.run(10)
    fig = plt.figure()
    ax = fig.add_subplot(122)
    bx = fig.add_subplot(121)

    vonoroi_diagram = Voronoi(centroids)
    voronoi_plot_2d(vonoroi_diagram, ax=ax, show_vertices=False, show_points=False)
    voronoi_plot_2d(vonoroi_diagram, ax=bx, show_vertices=False, show_points=False)

    for axis in fig.axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

    for cluster, cluster_weights, centroid in zip(
        clusters, clusters_weights, centroids
    ):
        ax.scatter(cluster[0], cluster[1], c=cluster_weights, cmap=plt.get_cmap("OrRd"))
        ax.plot(centroid[0], centroid[1], "kx", markersize=12)
        bx.scatter(cluster[0], cluster[1])
        bx.plot(centroid[0], centroid[1], "kx", markersize=12)

    plt.show()

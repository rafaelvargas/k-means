import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from clustering.KmeansClusterer import KmeansClusterer


def generate_random_points(number_of_points, dimension):
    points = np.random.rand(number_of_points, dimension)
    return points

if __name__ == '__main__':
    points = generate_random_points(5000, 3)
    clusterer = KmeansClusterer(5, points)
    clusters = clusterer.run(10)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cluster in clusters:
        ax.scatter(cluster[0], cluster[1], cluster[2])
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title("3D K-means Clustering")
    plt.show()
        
    

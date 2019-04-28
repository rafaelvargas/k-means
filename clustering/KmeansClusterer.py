import numpy as np
from tqdm import tqdm


class KmeansClusterer:
    def __init__(self, k, dataset):
        dataset_dimensions = dataset.shape
        self.number_of_clusters = k
        self.clusters = [[] for i in range(self.number_of_clusters)]
        self.dataset = dataset
        self.centroids = np.empty((self.number_of_clusters, dataset_dimensions[1]))
        random_points_indexes = np.random.choice(
            np.arange(dataset_dimensions[0]), k, replace=False
        )
        for i, rand in enumerate(random_points_indexes):
            self.centroids[i] = self.dataset[rand]

    def _assign_data_to_clusters(self):
        self.clusters = [[] for i in range(self.number_of_clusters)]
        for d in self.dataset:
            closest_centroid_index = 0
            smallest_distance = np.sqrt(np.sum(np.power(self.centroids[0] - d, 2)))
            for i, centroid in enumerate(self.centroids):
                distance = np.sqrt(np.sum(np.power(centroid - d, 2)))
                if distance < smallest_distance:
                    smallest_distance = distance
                    closest_centroid_index = i
            self.clusters[closest_centroid_index].append(d)

    def _update_centroids(self):
        for i, cluster in enumerate(self.clusters):
            self.centroids[i] = sum(cluster) / len(cluster)


    def run(self, number_of_iterations):
        for i in tqdm(range(number_of_iterations)):
            self._assign_data_to_clusters()
            self._update_centroids()
        
        result = []
        for cluster in self.clusters:
            result.append(np.array(cluster).T)
        
        return result

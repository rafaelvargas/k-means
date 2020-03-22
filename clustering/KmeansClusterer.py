import numpy as np
from tqdm import tqdm
from clustering.distance import distance_calculation_methods


class KmeansClusterer:
    def __init__(self, k, dataset, distance="euclidian"):
        dataset_dimensions = dataset.shape
        self.number_of_clusters = k
        self.clusters = [[] for i in range(self.number_of_clusters)]
        self.dataset = dataset
        try:
            self.distance_calculation_method = distance_calculation_methods[
                distance.lower()
            ]
        except KeyError:
            raise Exception("Invalid distance method: {}".format(distance))
        if (
            self.distance_calculation_method == "haversine"
            and dataset_dimensions[1] != 2
        ):
            raise Exception(
                "Invalid data set dimensions for the chosen distance calculation method"
            )
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
            smallest_distance = self._calculate_distance(self.centroids[0], d)
            for i, centroid in enumerate(self.centroids):
                distance = self._calculate_distance(centroid, d)
                if distance < smallest_distance:
                    smallest_distance = distance
                    closest_centroid_index = i
            self.clusters[closest_centroid_index].append(d)

    def _update_centroids(self):
        for i, cluster in enumerate(self.clusters):
            self.centroids[i] = sum(cluster) / len(cluster)

    def _calculate_distance(self, p1, p2):
        return self.distance_calculation_method(p1, p2)

    def run(self, number_of_iterations):
        for _ in tqdm(range(number_of_iterations)):
            self._assign_data_to_clusters()
            self._update_centroids()
        result = []
        for cluster in self.clusters:
            result.append(np.array(cluster).T)
        return result

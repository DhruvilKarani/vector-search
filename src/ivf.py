import numpy as np
import os
from sklearn.cluster import KMeans



class IVF_Flat:
    def __init__(self, X, topk, n_clusters=1000, n_probes=5):
        self.X = X
        self.n_clusters = n_clusters
        self.n_probes = n_probes
        self.codebook = None
        self.cluster_membership = None
        self.cluster_model = None
        self.topk = topk
        self._generate_codebook()
        self._generate_cluster_membership()
        self._validate_n_clusters()


    def _validate_n_clusters(self):
        assert self.n_clusters <= self.X.shape[0], "Number of clusters must be less than number of vectors"


    def _generate_codebook(self):
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(self.X)
        self.codebook = kmeans.cluster_centers_
        self.cluster_model = kmeans


    def _generate_cluster_membership(self):
        y_pred = self.cluster_model.predict(self.X)
        self.cluster_membership = {}
        cluster_indexes = np.arange(self.n_clusters)
        self.cluster_membership = {cluster: np.where(y_pred == cluster)[0].tolist() for cluster in cluster_indexes}
        self.cluster_model = None # We don't need the cluster model anymore


    def _compute_euclidean_distance(self, X, Y):
        return np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=-1)


    def _get_n_closest_clusters(self, X):
        distances = self._compute_euclidean_distance(X, self.codebook)
        closest_cluster_idxes = np.argsort(distances, axis=-1)[:, :self.n_probes].reshape(-1)
        return closest_cluster_idxes
    

    def _get_candidates(self, probing_cluster_indexes):
        candidates = []
        for cluster_idx in probing_cluster_indexes:
            candidates.extend(self.cluster_membership[cluster_idx])
        return candidates


    def __call__(self, query):
        assert query.shape == (1, self.X.shape[1]), "Query shape is not correct. Expected: {}, Got: {}".format(self.X.shape[1], query.shape[1])
        probing_cluster_indexes = self._get_n_closest_clusters(query)
        candidates = self._get_candidates(probing_cluster_indexes)
        candidate_vectors = self.X[candidates]
        candidate_distances = self._compute_euclidean_distance(query, candidate_vectors).flatten()
        topk_idxes = [candidates[i] for i in np.argsort(candidate_distances, axis=-1)[:self.topk]]
        topk_distances = candidate_distances[topk_idxes]
        return topk_idxes, topk_distances



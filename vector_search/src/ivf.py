import numpy as np
import os
from sklearn.cluster import KMeans
from typing import List


class IVF_Flat:
    def __init__(self, X: np.ndarray, topk: int, n_clusters :int = 1000, n_probes :int = 5):
        self.X = X
        self.n_clusters = n_clusters
        self.n_probes = n_probes
        self.codebook = None
        self.cluster_membership = None
        self.cluster_model = None
        self.topk = topk
        self._generate_codebook()
        self.normalized_codebook = self.codebook/np.linalg.norm(self.codebook, axis=-1)[:, np.newaxis]
        self._generate_cluster_membership()
        self._validate_n_clusters()


    def _validate_n_clusters(self) -> None:
        assert self.n_clusters <= self.X.shape[0], "Number of clusters must be less than number of vectors"


    def _generate_codebook(self) -> None:
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(self.X)
        self.codebook = kmeans.cluster_centers_
        self.cluster_model = kmeans


    def _generate_cluster_membership(self) -> None:
        y_pred = self.cluster_model.predict(self.X)
        self.cluster_membership = {}
        cluster_indexes = np.arange(self.n_clusters)
        self.cluster_membership = {cluster: np.where(y_pred == cluster)[0].tolist() for cluster in cluster_indexes}
        self.cluster_model = None # We don't need the cluster model anymore


    def _compute_cosine_distance(self, X, Y) -> np.ndarray:
        return 1-np.matmul(X, Y.T)


    def _get_n_closest_clusters(self, X) -> List[int]:
        distances = self._compute_cosine_distance(X, self.normalized_codebook)
        closest_cluster_idxes = np.argsort(distances, axis=-1)[:, :self.n_probes].reshape(-1)
        return closest_cluster_idxes
    

    def _get_candidates(self, probing_cluster_indexes) -> List[int]:
        candidates = []
        for cluster_idx in probing_cluster_indexes:
            candidates.extend(self.cluster_membership[cluster_idx])
        return candidates


    def __call__(self, query) -> List[int]:
        query = query/np.linalg.norm(query)
        assert query.shape == (1, self.X.shape[1]), "Query shape is not correct. Expected: {}, Got: {}".format(self.X.shape[1], query.shape[1])
        probing_cluster_indexes = self._get_n_closest_clusters(query)
        candidates = self._get_candidates(probing_cluster_indexes)
        candidate_vectors = self.X[candidates]
        candidate_distances = self._compute_cosine_distance(query, candidate_vectors).flatten()
        candidate_idxes = np.argsort(candidate_distances, axis=-1)[:self.topk]
        topk_idxes = [candidates[i] for i in candidate_idxes]
        topk_distances = candidate_distances[candidate_idxes]
        return topk_idxes, topk_distances




class IVF_SQ8:
    def __init__(self, X: np.ndarray, topk: int, n_clusters: int = 1000, n_probes: int = 5):
        self.X = X
        self.n_clusters = n_clusters
        self.n_probes = n_probes
        self.codebook = None
        self.cluster_membership = None
        self.cluster_model = None
        self.topk = topk
        self._generate_codebook()
        self.normalized_codebook = self.codebook/np.linalg.norm(self.codebook, axis=-1)[:, np.newaxis]
        self._generate_cluster_membership()
        self._validate_n_clusters()
        self._set_quantization_parameters()
        self.quantized_X = self.quantize_vectors(self.X)

    def _validate_n_clusters(self) -> None:
        assert self.n_clusters <= self.X.shape[0], "Number of clusters must be less than number of vectors"


    def _generate_codebook(self) -> None:
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(self.X)
        self.codebook = kmeans.cluster_centers_
        self.cluster_model = kmeans


    def _generate_cluster_membership(self) -> None:
        y_pred = self.cluster_model.predict(self.X)
        self.cluster_membership = {}
        cluster_indexes = np.arange(self.n_clusters)
        self.cluster_membership = {cluster: np.where(y_pred == cluster)[0].tolist() for cluster in cluster_indexes}
        self.cluster_model = None # We don't need the cluster model anymore


    def _compute_proxy_euclidean_distance(self, X, Y) -> np.ndarray:
        return np.square(X[:, np.newaxis] - Y[np.newaxis, :]).sum(axis=-1)


    def _get_n_closest_clusters(self, X) -> List[int]:
        distances = self._compute_proxy_euclidean_distance(X, self.normalized_codebook)
        closest_cluster_idxes = np.argsort(distances, axis=-1)[:, :self.n_probes].reshape(-1)
        return closest_cluster_idxes
    

    def _get_candidates(self, probing_cluster_indexes) -> List[int]:
        candidates = []
        for cluster_idx in probing_cluster_indexes:
            candidates.extend(self.cluster_membership[cluster_idx])
        return candidates


    def _set_quantization_parameters(self) -> None:
        self.quantization_bins = 2**8
        self.min = self.X.min()
        self.max = self.X.max()
        self.range = self.max - self.min


    def quantize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        normalized_vectors = (vectors - self.min)/(self.range)
        quantized_vectors = np.floor(normalized_vectors*self.quantization_bins)
        quantized_vectors = np.clip(quantized_vectors, a_max=255, a_min=0)
        quantized_vectors = quantized_vectors.astype(np.uint8)
        return quantized_vectors


    def __call__(self, query) -> List[int]:
        query = query/np.linalg.norm(query)
        assert query.shape == (1, self.X.shape[1]), "Query shape is not correct. Expected: {}, Got: {}".format(self.X.shape[1], query.shape[1])
        probing_cluster_indexes = self._get_n_closest_clusters(query)
        candidates = self._get_candidates(probing_cluster_indexes)
        candidate_vectors = self.quantized_X[candidates]
        query = self.quantize_vectors(query)
        candidate_distances = self._compute_proxy_euclidean_distance(query, candidate_vectors).flatten()
        candidate_idxes = np.argsort(candidate_distances, axis=-1)[:self.topk]
        topk_idxes = [candidates[i] for i in candidate_idxes]
        topk_distances = candidate_distances[candidate_idxes]
        return topk_idxes, topk_distances




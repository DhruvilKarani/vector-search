import numpy as np


class ExactSearch:
    def __init__(self, X, topk):
        self.X = X
        self.topk = topk

    def _compute_cosine_distance(self, Y):
        return 1-np.matmul(self.X, Y.T)

    def __call__(self, query):
        query = query/np.linalg.norm(query)
        candidate_distances = self._compute_cosine_distance(query).flatten()
        topk_idxes = np.argsort(candidate_distances, axis=-1)[:self.topk]
        topk_distances = candidate_distances[topk_idxes]
        return topk_idxes, topk_distances



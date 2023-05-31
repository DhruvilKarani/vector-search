import sys
from vector_search.src.ivf import IVF_Flat
from vector_search.data.utils import generate_dummy_data
import numpy as np
import pytest

@pytest.fixture(scope="session")
def data():
    base_size = 1000
    query_size = 1
    dim = 128
    base, queries, n_queries_idxes = generate_dummy_data(base_size, query_size, dim)
    base = base/np.linalg.norm(base, axis=-1)[:, np.newaxis]
    return base, queries, n_queries_idxes


@pytest.fixture(scope="session")
def identity_data():
    query_size = 1
    dim = 128
    base_size = dim
    base = np.identity(dim).astype(np.float32)
    n_queries_idxes = np.random.choice(base_size, query_size, replace=False)
    queries = base[n_queries_idxes]
    base = base/np.linalg.norm(base, axis=-1)[:, np.newaxis]
    return base, queries, n_queries_idxes


def test_ivf_flat_construction(data):
    ivf = IVF_Flat(data[0], topk=3)


def test_codebook_generation(data):
    ivf = IVF_Flat(data[0], topk=3)
    ivf._generate_codebook()
    assert ivf.codebook.shape == (ivf.n_clusters, data[0].shape[1]), "Codebook shape is not correct"


def test_get_n_closest_clusters(identity_data):
    ivf = IVF_Flat(identity_data[0], n_clusters=128, topk=10)
    ivf.normalized_codebook = identity_data[0]
    closest_clusters = ivf._get_n_closest_clusters(identity_data[1])
    assert np.equal(closest_clusters[:1], identity_data[2]).all(), "Closest clusters are not correct"


def test_cosine_distance():
    X = np.array([[1, 0, 0], [0, 0, 1]])
    Y = np.array([[1, 0, 0], [0, 0, 1]])
    ivf = IVF_Flat(X, topk=3, n_clusters=2)
    distances = ivf._compute_cosine_distance(X, Y)
    assert np.isclose(distances, np.array([[0, 1], [1, 0]])).all(), "Euclidean distance is not correct"


@pytest.mark.parametrize("n_clusters", [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_ivf_flat_exact_search(n_clusters):
    base_size = 10000
    query_size = 1
    dim = 32

    base = np.random.rand(base_size, dim).astype(np.float32)
    base = base/np.linalg.norm(base, axis=-1)[:, np.newaxis]
    n_query_idxes = np.random.choice(base_size, query_size, replace=False)
    query = base[n_query_idxes]
    print(n_query_idxes)
    ivf = IVF_Flat(base, n_clusters=n_clusters, topk=1, n_probes=n_clusters)

    idx, _ = ivf(query)
    assert idx[0] == n_query_idxes[0], "Result is not correct. Expected: {}, Actual: {}".format(n_query_idxes[0], idx[0])


@pytest.mark.parametrize("n_clusters, n_probes", [(2, 2), (3, 3), (14, 4), (50, 5), (46, 6), (100, 7), (100, 1), (500, 1)])
def test_recall_at_one(data, n_clusters, n_probes):
    base, queries, n_queries_idxes = data
    ivf = IVF_Flat(base, topk=1, n_clusters=n_clusters, n_probes=n_probes)
    
    score = 0
    for query, ground_truth in zip(queries, n_queries_idxes):
        query = query.reshape(1, -1)
        idxes, _ = ivf(query)
        if ground_truth in idxes:
            score += 1
    recall = score / len(n_queries_idxes)
    assert recall == 1.0, "Recall@1 is not 1.0. Expected: 1.0, Actual: {}".format(recall)

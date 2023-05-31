import pytest
import numpy as np
from vector_search.data.utils import generate_dummy_data


def test_generate_dummy_data():
    base_size = 1000
    query_size = 10
    dim = 128
    base, queries, n_queries_idxes = generate_dummy_data(base_size, query_size, dim)
    assert np.equal(base[n_queries_idxes], queries).all()
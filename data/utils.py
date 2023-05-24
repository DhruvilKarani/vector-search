import numpy as np

def generate_dummy_data(n_base, n_queries, n_dim):
    base = np.random.rand(n_base, n_dim).astype(np.float32)
    assert n_base > n_queries, "n_base must be greater than n_neighbours"
    n_queries_idxes = np.random.choice(n_base, n_queries, replace=False)
    queries = base[n_queries_idxes]
    return base, queries, n_queries_idxes

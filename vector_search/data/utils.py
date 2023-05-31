import numpy as np
import json

def generate_dummy_data(n_base, n_queries, n_dim):
    base = np.random.rand(n_base, n_dim).astype(np.float32)
    assert n_base > n_queries, "n_base must be greater than n_neighbours"
    n_queries_idxes = np.random.choice(n_base, n_queries, replace=False)
    queries = base[n_queries_idxes]
    return base, queries, n_queries_idxes

def load_glove_data():
    directory = "/Users/dhruvilkarani/Desktop/substack/vector-search-engine/vectors/glove"
    words = open(directory + "/words.txt").read().splitlines()
    X = np.load(directory + "/X.npy")
    assert len(words) == X.shape[0], "Number of words and number of vectors are not equal"
    similar_word_mapping = json.load(open(directory + "/similar_word_mapping.json"))
    return words, X, similar_word_mapping
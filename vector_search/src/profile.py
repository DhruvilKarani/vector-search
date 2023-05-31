from ivf import IVF_Flat, IVF_SQ8
import numpy as np
import sys
sys.path.append('..')
from data.utils import generate_dummy_data, load_glove_data
from tqdm import tqdm
from baseline import ExactSearch
import time

def compute_scores(index):
    score = 0
    query_time = []
    for query, ground_truth in tqdm(zip(QUERIES, GROUND_TRUTH_IDXES)):
        query = query.reshape(1, -1)
        start_time = time.time()
        idxes, _ = index(query)
        end_time = time.time()
        query_time.append(end_time - start_time)
        if ground_truth in idxes:
            score += 1
    recall = score / len(QUERIES)
    rate = np.mean(query_time)
    print("Average query time: {}".format(rate))
    print("Accuracy: {}".format(recall))
    return recall, rate


class ProfileConfig:
    TOPK = 3
    N_CLUSTERS = 1000
    N_PROBES = 10
    N_QUERIES = 5000

words, X, similar_word_mapping = load_glove_data()
X_normalized = X/np.linalg.norm(X, axis=-1)[:, np.newaxis]
random_idxes = np.random.choice(len(words), ProfileConfig.N_QUERIES, replace=False)
QUERIES = X_normalized[random_idxes]
SAMPLE_WORDS = [words[idx] for idx in random_idxes]
GROUND_TRUTH_IDXES = [words.index(similar_word_mapping[word]) for word in SAMPLE_WORDS]


print("building IVF-Flat...")
ivf_flat = IVF_Flat(X_normalized, topk=ProfileConfig.TOPK, n_clusters=ProfileConfig.N_CLUSTERS, n_probes=ProfileConfig.N_PROBES)
print("building IVF-SQ8...")
ivf_sq8 = IVF_SQ8(X_normalized, topk=ProfileConfig.TOPK, n_clusters=ProfileConfig.N_CLUSTERS, n_probes=ProfileConfig.N_PROBES)
print("building Exact search...")
exact = ExactSearch(X_normalized, topk=ProfileConfig.TOPK)




if __name__ == "__main__":

    print("IVF-SQ8")
    compute_scores(ivf_sq8)
    print("IVF-Flat")
    compute_scores(ivf_flat)
    print("Exact")
    compute_scores(exact)
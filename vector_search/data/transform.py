import os
import numpy as np
import json
from tqdm import tqdm
DIRECTORY = "../../vectors/glove"

def read_glove(directory=DIRECTORY):

    #read list of words from os.path.join(directory, "words.txt")
    words = []
    with open(os.path.join(directory, "words.txt"), "r") as f:
        words = f.read().splitlines()
    
    #read numpy array X.npy
    X = np.load(os.path.join(directory, "X.npy"))
    assert len(words) == X.shape[0], "Number of words and number of vectors are not equal"
    return words, X


def find_most_similar_cosine(word, words, X_normalized, topk=1):
    #find the index of the word
    idx = words.index(word)
    #get the vector representation of the word
    query = X_normalized[idx]
    #compute the cosine similarity using query and X_normalized
    cosine_similarity = np.dot(query, X_normalized.T)
    #find topk most similar words
    topk_idxes = np.argsort(cosine_similarity)[::-1][:topk]
    topk_words = [words[idx] for idx in topk_idxes]
    return topk_words


if __name__ == '__main__':
    words, X = read_glove()
    X_normalized = X/np.linalg.norm(X, axis=-1)[:, np.newaxis]

    similar_word_mapping = {}
    #find 2nd most similar word to each word in the word list
    for word in tqdm(words):
        similar_word = find_most_similar_cosine(word, words, X_normalized, topk=2)[1]
        similar_word_mapping[word] = similar_word
    
    #save simiar_word_mapping as json using json.dump
    with open(os.path.join(DIRECTORY, "similar_word_mapping.json"), "w") as f:
        json.dump(similar_word_mapping, f)





import os
import numpy as np

def read_sift(directory="../vectors/sift"):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith("vecs")]
    data_list = []
    for file in files:
        print(file)
        data = np.fromfile(file, dtype=np.float32)
        if file.endswith("fvecs"):
            vector_length = 129 
            num_vectors = len(data) // vector_length
            data = data.reshape(num_vectors, vector_length)
            data_list.append(data)
        else:
            data_list.append(data)
        print(data.shape)
    return data_list

if __name__ == '__main__':
    arrays = read_sift()


import os
import numpy as np

# if os.path.exists('../vectors'):
#     os.system('rm -rf ../vectors')
# os.system('mkdir ../vectors')

# # download ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz in ../vectors using wget
# os.system('wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz -P ../vectors')
# os.system('tar -xzf ../vectors/sift.tar.gz -C ../vectors')


def _load_texmex_vectors(f, n, k):
    import struct

    v = np.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack("f" * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t, fn):
    import struct

    m = t.getmember(fn)
    f = t.extractfile(m)
    (k,) = struct.unpack("i", f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift():
    import tarfile
    fn = os.path.join("../vectors", "sift.tar.gz")
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "sift/sift_base.fvecs")
        test = _get_irisa_matrix(t, "sift/sift_query.fvecs")
        groundtruth = _get_irisa_matrix(t, "sift/sift_groundtruth.ivecs")
        learn = _get_irisa_matrix(t, "sift/sift_learn.fvecs")
    
    return train, test, groundtruth, learn

if __name__ == "__main__":
    train, test, groundtruth, learn = sift()
    print(train.shape)
    print(test.shape)
    print(groundtruth.shape)
    print(learn.shape)
    print(train[0])
    print(test[0])
    print(groundtruth[0])
    print(learn[0])
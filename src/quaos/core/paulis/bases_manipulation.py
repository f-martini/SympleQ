import numpy as np


def bases_to_int(aa, dims):
    dims = np.flip(dims)
    aa = np.flip(aa)
    a = aa[0] + sum([aa[i1] * np.prod(dims[:i1]) for i1 in range(1, len(dims))])
    dims = np.flip(dims)
    aa = np.flip(aa)
    return a

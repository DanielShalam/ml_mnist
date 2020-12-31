import numpy as np


def getEigen(images):
    # compute mean
    mu = np.mean(images, axis=0)
    # subtract mean
    Z = images - mu
    # compute scatter matrix
    S = np.matmul(Z.transpose(), Z)
    # compute sorted eigen values and vectors
    d, v = np.linalg.eig(S)

    return d, v, Z

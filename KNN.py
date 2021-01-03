import numpy as np


def performKnnClassification(k, validation_images, train_images, train_labels):
    distances = []
    labels = []

    # compute euclidean distance matrix between each image in the validation to each image in the training set
    dist_matrix = distance_matrix(validation_images, train_images)

    # for each row in the distance matrix, we will take the k smallest distances to get the k neighbours
    for row in dist_matrix:
        min_indices = np.argpartition(row, k)
        distances.append(row[min_indices[:k]])
        labels.append(train_labels[min_indices[:k]])

    return distances, labels


def getClassification(neighbors_labels):
    (values, counts) = np.unique(neighbors_labels, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]


def distance_matrix(A, B, squared=False):
    """
    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    See also
    --------
    A more generalized version of the distance matrix is available from
    scipy (https://www.scipy.org) using scipy.spatial.distance_matrix,
    which also gives a choice for p-norm.
    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(M, 1))
    D_squared = A_dots + B_dots - 2 * A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared

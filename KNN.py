import numpy as np


def performKnnClassification(k, test_images, train_images, train_labels):
    distances = []
    labels = []

    # compute euclidean distance matrix between each image in the validation to each image in the training set
    dist_matrix = computeDistanceMatrix(test_images, train_images)

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


def computeDistanceMatrix(A, B):
    # Compute all pairwise distances between vectors in A and B.
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(M, 1))
    D_squared = A_dots + B_dots - 2 * A.dot(B.T)

    zero_mask = np.less(D_squared, 0.0)
    D_squared[zero_mask] = 0.0
    return np.sqrt(D_squared)

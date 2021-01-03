import numpy as np
import math


def performKnnClassification(k, validation_images, train_images, train_labels):
    distances = []
    labels = []
    
    neighbors_dist = np.zeros(k, dtype=float)
    neighbors_labels = np.zeros(k, dtype=int)

    for v_image in validation_images:
        max_dist = [-1, math.inf]  # idx, value of the largest distance
        for idx, (t_image, t_label) in enumerate(zip(train_images, train_labels)):
            distance = np.linalg.norm(v_image - t_image)
            if idx < k:
                neighbors_dist[idx] = distance
                neighbors_labels[idx] = t_label
                if max_dist[1] > distance:
                    max_dist[0] = idx
                    max_dist[1] = distance

            else:
                if max_dist[1] > distance:
                    neighbors_dist[max_dist[0]] = distance
                    neighbors_labels[max_dist[0]] = t_label
                    max_dist[0] = np.argmax(neighbors_dist)
                    max_dist[1] = neighbors_dist[max_dist[0]]

        # classifications.append(getClassification(neighbors_labels))
        distances.append(neighbors_dist.copy())
        labels.append(neighbors_labels.copy())

    return distances, labels


def getClassification(neighbors_labels):
    (values, counts) = np.unique(neighbors_labels, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

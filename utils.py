import matplotlib.pyplot as plt
import numpy as np
import math


def computeCentroids(digits, images, labels, plot_centroids):
    """ step2: function to compute centroids for each digit. """
    centroids = dict.fromkeys(digits)
    num_samples = dict.fromkeys(digits, 0)

    for digit in digits:
        centroids[digit] = np.zeros(len(images[0]))

    # compute sum of all samples per class and count them
    for i, label in enumerate(labels):
        centroids[label] += images[i]
        num_samples[label] += 1

    for digit in digits:
        centroids[digit] /= num_samples[digit]

    if plot_centroids: plotCentroids(centroids)
    return centroids


def computeCentroidsDist(digits, centroids):
    """ step3: function to compute the euclidean distance between each pair of digits centroids. """
    max_digit = max(digits) + 1
    dist_matrix = np.zeros((max_digit, max_digit))
    centroids_array = []
    for d in digits:
        centroids_array.append(centroids[d])

    for first_digit in digits:
        for sec_digit in digits:
            if first_digit == sec_digit:
                dist_matrix[first_digit][first_digit] = 0

            # Euclidean distance
            dist_matrix[first_digit][sec_digit] = math.sqrt(
                sum([(a - b) ** 2 for a, b in zip(centroids[first_digit], centroids[sec_digit])]))

    from tabulate import tabulate

    headers = list(centroids.keys())

    # tabulate data
    table = tabulate(dist_matrix, headers, showindex=True, tablefmt="fancy_grid")
    print("Euclidean distance Matrix: ")
    print(table)

    return dist_matrix


def computePixelVar(images):
    """ step4: function to compute the variance of each pixel for all given images. """
    num_pixels = len(images[0])
    variance_vec = np.zeros(num_pixels)

    for pixel in range(num_pixels):
        variance_vec[pixel] = np.var(images[:, pixel])

    # now we will split the variance vector to 10 equal size intervals and count the result
    # and plot histogram
    _ = plt.hist(variance_vec, bins=10)
    plt.title("Histogram of Pixels variance")
    plt.show()

    return variance_vec


def plotClassImage(digits, images, labels):
    """ step1: function to plot 1 image for each class in the dataset. """
    for digit in digits:
        for i, label in enumerate(labels):
            if label == digit:
                # Plot image
                image = np.reshape(np.asarray(images[i]), [28, 28])
                plt.imshow(image)
                plt.savefig(f'plotImages/sample_digit-{digit}.png')
                plt.show()
                break

    plt.close()


def plotCentroids(centroids):
    """ function to plot and save centroids images. """
    for digit in centroids.keys():
        image = np.reshape(np.asarray(centroids[digit]), [28, 28])
        plt.imshow(image)
        plt.savefig(f'plotImages/centroid_digit-{digit}.png')
        plt.show()

    plt.close()

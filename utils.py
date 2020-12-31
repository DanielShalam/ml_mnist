import matplotlib.pyplot as plt
import numpy as np


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


def computeDistMatrix(digits, centroids):
    """ step3: function to compute the euclidean distance between each pair of digits centroids. """
    max_digit = max(digits) + 1
    dist_matrix = np.zeros((max_digit, max_digit))

    for first_digit in digits:
        for sec_digit in digits:

            if first_digit >= sec_digit:
                dist_matrix[first_digit - 1][sec_digit - 1] = dist_matrix[sec_digit - 1][first_digit - 1]
                continue

            dist_matrix[first_digit - 1][sec_digit - 1] = np.linalg.norm(centroids[first_digit] - centroids[sec_digit])

    from tabulate import tabulate

    headers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

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


def plotPCA(y):
    marker_size = 15
    plt.scatter(y[0][0:100], y[1][0:100], marker_size, c='r', label='0')
    plt.scatter(y[0][100:200], y[1][100:200], marker_size, c='y', label='1')
    plt.scatter(y[0][200:300], y[1][200:300], marker_size, c='g', label='2')
    plt.scatter(y[0][300:400], y[1][300:400], marker_size, c='b', label='3')
    plt.scatter(y[0][400:500], y[1][400:500], marker_size, c='c', label='4')
    plt.scatter(y[0][500:600], y[1][500:600], marker_size, c='purple', label='5')
    plt.scatter(y[0][600:700], y[1][600:700], marker_size, c='turquoise', label='6')
    plt.scatter(y[0][700:800], y[1][700:800], marker_size, c='orange', label='7')
    plt.scatter(y[0][800:900], y[1][800:900], marker_size, c='salmon', label='8')
    plt.scatter(y[0][900:1000], y[1][900:1000], marker_size, c='brown', label='9')

    plt.title("Point observations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(title="Classes")
    plt.show()
    plt.close()

    plt.scatter(y[0][0:100], y[1][0:100], marker_size, c='r', label='0')
    plt.scatter(y[0][100:200], y[1][100:200], marker_size, c='b', label='1')
    plt.scatter(y[0][900:1000], y[1][900:1000], marker_size, c='g', label='9')

    plt.title("Point observations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(title="Classes")
    plt.show()
    plt.close()

# Loading packages
from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import os
import threading

# Loading modules
import utils
import PCA
import KNN


class myThread(threading.Thread):
    def __init__(self, threadID, k, val, train, labels):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.k = k
        self.val = val
        self.train = train
        self.t_labels = labels
        self.distances = None
        self.labels = None

    def run(self):
        self.distances, self.labels = KNN.performKnnClassification(k, self.val, self.train, self.t_labels)


CEND = '\33[0m'
CRED = '\33[31m'
CGREEN = '\33[32m'

threads = []

if __name__ == '__main__':
    plot_centroids = False
    mndata = MNIST(os.path.join('mnistDataset'))

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    digits = set(train_labels)

    #### step 1 ####
    # utils.plotClassImage(digits, train_images, train_labels)

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    # test_labels = np.array(test_labels)

    #### step 2 ####
    centroids = utils.computeCentroids(digits, train_images, train_labels, plot_centroids)

    #### step 3 ####
    dist_matrix = utils.computeDistMatrix(digits, centroids)

    #### step 4 ####
    variance_vec = utils.computePixelVar(np.concatenate((train_images, test_images), axis=0))

    #### step 5 ####
    # remove indices with 0 variance
    indices = np.where(0 == variance_vec)
    new_train_images = np.delete(train_images, obj=indices, axis=1)
    new_test_images = np.delete(test_images, obj=indices, axis=1)

    ########### PCA ###########
    k = 2
    d, v, Z = PCA.getEigen(new_train_images)

    E = np.reshape(v[:][0:k], [k, new_train_images.shape[1]])

    indices = []
    for digit in digits:
        indices.extend(np.where(digit == train_labels)[0][:100])

    y = np.matmul(E, Z[indices].transpose())

    PCA.plotPCA(y)

    #### step 6 ####
    e_sum = sum(d)
    temp_sum = 0
    cum_pct = []

    for e_value in d:
        temp_sum += e_value
        cum_pct.append(temp_sum / e_sum * 100)

    plt.plot(np.arange(len(cum_pct)), cum_pct)
    plt.grid()

    plt.title("Energy in % for each value, from total value")
    plt.show()
    plt.close()
    ################

    temp_sum = 0
    energy = 0.9
    t_e_sum = e_sum * energy
    k = 0
    for e_value in d:
        if t_e_sum <= temp_sum:
            break

        temp_sum += e_value
        k += 1

    print(f"\nThe chosen k is: {k}, which is {str(temp_sum / e_sum * 100)[:2]}% of the Energy")

    #### step 7 ####
    # KNN
    # PCA to k dimensions for each feature vector
    # TODO: delete those rows
    # new_train_images = new_train_images[:5000]
    # train_labels = train_labels[:5000]

    E = np.reshape(v[:][0:k], [k, new_train_images.shape[1]])
    y = np.matmul(E, Z.transpose())
    y = y.transpose()

    # split train-validation
    validation_pct = 0.15
    val_idx = int(y.shape[0] * validation_pct)
    validation_images = y[:val_idx]         # validation set
    validation_labels = train_labels[:val_idx]
    new_train_images = y[val_idx:]          # train set without validation set
    train_labels = train_labels[val_idx:]

    # perform knn
    k = 10
    classifications = []
    num_threads = 5
    num_samples = len(new_train_images)
    part_duration = num_samples / num_threads
    ranges = [(i * part_duration, (i + 1) * part_duration) for i in range(num_threads)]

    import math
    for i in range(num_threads):
        start = math.floor(ranges[i][0])
        end = math.floor(ranges[i][1])
        if i == num_threads: end += 1
        thread = myThread(i, k, validation_images, new_train_images[start:end], train_labels[start:end]) # Create new threads
        thread.start()          # Start new Thread
        threads.append(thread)  # Add threads to thread list

    # Wait for all threads to complete
    for t in threads:
        t.join()

    for i in range(len(validation_images)):
        distances = []
        labels = []
        for t in threads:   # for each thread we will merge its neighbours with all the rest
            distances.extend(t.distances[i])
            labels.extend(t.labels[i])

        indices = (np.asarray(distances)).argsort()[:k]    # getting final array
        final_labels = [labels[j] for j in indices]
        classifications.append(KNN.getClassification(final_labels))

    true_count = 0
    for i in range(len(classifications)):
        if classifications[i] == validation_labels[i]: true_count += 1

    print(f"{CGREEN}\nClassification success: {true_count / len(classifications)*100}%{CEND}")

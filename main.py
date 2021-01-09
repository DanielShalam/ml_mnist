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
    def __init__(self, threadID, curr_k, val, train, t_labels):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.k = curr_k
        self.val = val
        self.train = train
        self.t_labels = t_labels
        self.distances = None
        self.labels = None
        return

    def run(self):
        self.distances, self.labels = KNN.performKnnClassification(k, self.val, self.train, self.t_labels)
        return


def multiThreadedKNN(x, Y, x_labels, y_labels, k):
    threads = []
    classifications = []
    num_samples = len(Y)
    part_duration = num_samples / num_threads
    ranges = [(i * part_duration, (i + 1) * part_duration) for i in range(num_threads)]

    import math

    for i in range(num_threads):
        start = math.floor(ranges[i][0])
        end = math.floor(ranges[i][1])
        if i == num_threads: end += 1
        thread = myThread(i, k, x, Y[start:end], y_labels[start:end])  # Create new threads
        thread.start()  # Start new Thread
        threads.append(thread)  # Add threads to thread list

    # Wait for all threads to complete
    for t in threads:
        t.join()

    for i in range(len(x)):
        distances = []
        labels = []
        for t in threads:  # for each thread we will merge its neighbours with all the rest
            distances.extend(t.distances[i])
            labels.extend(t.labels[i])

        # getting the k best neighbours from all threads
        # assign k best labels
        if num_threads > 1:
            min_indices = np.argpartition(distances, k)[:k]
            final_labels = [labels[j] for j in min_indices]
        else:
            final_labels = labels

        classifications.append(KNN.getClassification(final_labels))

    true_count = 0
    if len(classifications) != len(x_labels):
        print("Length of classification list not equal to the number of labels.")
        return 0

    for i in range(len(classifications)):
        if classifications[i] == x_labels[i]: true_count += 1

    accuracy = true_count / len(classifications) * 100
    print(f"{CGREEN}Classification accuracy: {accuracy}%{CEND}")

    return accuracy


CEND = '\33[0m'
CRED = '\33[31m'
CGREEN = '\33[32m'

threads = []
num_threads = 5

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
    d, v, Z, mu = PCA.getEigen(new_train_images)

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

    print(f"{CGREEN}\nThe chosen k is: {k}, which is {str(temp_sum / e_sum * 100)[:2]}% of the Energy.{CEND}")

    #### step 7 ####
    # KNN
    n = k
    # PCA to k dimensions for each feature vector
    # TODO should we do PCA separately to validation set
    E = np.reshape(v[:][0:n], [n, new_train_images.shape[1]])
    y = np.matmul(E, Z.transpose())
    y = y.transpose()

    # split train-validation
    validation_pct = 0.15
    val_idx = int(y.shape[0] * validation_pct)
    validation_images = y[:val_idx]  # validation set
    validation_labels = train_labels[:val_idx]
    new_train_images = y[val_idx:]  # train set without validation set
    t_labels = train_labels[val_idx:]

    # perform knn
    k_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 30]
    best_k = [-1, -1]  # accuracy, k
    accuracy_list = []

    for k_idx, k in enumerate(k_list):
        threads = []
        classifications = []
        import math

        print(f"\nPerform KNN to validation set, K = {k}: ")
        accuracy = multiThreadedKNN(x=validation_images, Y=new_train_images, x_labels=validation_labels, y_labels=t_labels, k=k)

        accuracy_list.append(accuracy)
        if best_k[0] < accuracy:
            best_k[0] = accuracy
            best_k[1] = k

    plt.plot(k_list, accuracy_list)
    plt.show()

    ###### perform KNN to test set using the best k ######
    #### perform PCA to test set #####
    # PCA to n dimensions for each feature vector
    train_set = y
    Z = new_test_images - mu  # subtract the mean of the training set
    E = np.reshape(v[:][0:n], [n, new_test_images.shape[1]])
    y = np.matmul(E, Z.transpose())
    y = y.transpose()

    k = best_k[1]
    print(f"\nPerform KNN to test set, K = {k}: ")
    multiThreadedKNN(x=y, Y=train_set, x_labels=test_labels, y_labels=train_labels, k=k)

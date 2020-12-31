# Loading packages
from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import os

# Loading modules
import utils
import PCA

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
    
    utils.plotPCA(y)

    #### step 6 ####
    e_sum = sum(d)
    temp_sum = 0
    cum_pct = []

    for e_value in d:
        temp_sum += e_value
        cum_pct.append(temp_sum/e_sum*100)

    plt.plot(np.arange(len(cum_pct)), cum_pct)
    plt.grid()

    plt.title("Energy in % for each value, from total value")
    plt.show()
    plt.close()
    ################

    temp_sum = 0
    energy = 0.9
    t_e_sum = e_sum*energy
    k = 0
    for e_value in d:
        if t_e_sum <= temp_sum:
            break

        temp_sum += e_value
        k += 1

    print(f"The chosen k is: {k} which is {str(temp_sum/e_sum*100)[:2]}% of the Energy")

    #### step 7 ####
    # KNN
    # PCA to k dimensions for each feature vector
    E = np.reshape(v[:][0:k], [k, new_train_images.shape[1]])
    y = np.matmul(E, Z.transpose())
    y = y.transpose()
    
    # split train-validation
    validation_pct = 0.15
    val_idx = int(y.shape[0]*validation_pct)
    validation_images = y[:val_idx]
    validation_labels = train_labels[:val_idx]
    new_train_images = y[val_idx:]
    train_labels = train_labels[val_idx:]
    


import numpy as np
import matplotlib.pyplot as plt

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

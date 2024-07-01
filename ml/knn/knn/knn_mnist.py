from dataload import load_mnist_data

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import numpy as np
import time


def test_neibours():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    print(indices)

def calc_gamma(X,Y):
    num = 8
    nbrs = NearestNeighbors(n_neighbors=num, algorithm='ball_tree').fit(X)
    _,inds = nbrs.kneighbors(X)
    # print(inds)
    for i in range(inds.shape[0]):
        for j in range(inds.shape[1]):
            k = inds[i,j]
            inds[i,j] = Y[k]



    gammas = np.zeros((inds.shape[0],10),dtype=int)

    for i in range(inds.shape[0]):
        for j in range(inds.shape[1]):
            gammas[i][inds[i,j]] += 1

    return gammas

def calc_indents(gammas,Y):
    gamma_xiyi = np.zeros((gammas.shape[0],), dtype=int)
    gamma_xiy = np.zeros((gammas.shape[0],), dtype=int)
    for i in range(gammas.shape[0]):
        gamma_xiyi[i] = gammas[i,Y[i]]
    for i in range(gammas.shape[0]):
        a = gammas[i,:Y[i]]
        b = gammas[i,Y[i]+1:]
        amax = -1
        bmax = -1
        if len(a) != 0:
            amax = a.max()
        else:
            amax = b.max()
        if len(b) != 0:
            bmax = b.max()
        else:
            bmax = a.max()

        gamma_xiy[i] = max(amax,bmax)
        # print(amax,bmax,max(amax,bmax))
    # k = 4
    # print(Y[:k])
    # print(gammas[:k])
    # print(gamma_xiyi[:k])
    # print(gamma_xiy[:k])
    return gamma_xiyi - gamma_xiy

def experiment():
    path = "../../data/MNIST_dataset/mnist_train.csv"
    path_test = "../../data/MNIST_dataset/mnist_test.csv"
    X_train, Y_train = load_mnist_data(path)
    X_test, Y_test = load_mnist_data(path_test)
    Y_train = Y_train.argmax(axis=1)
    Y_test = Y_test.argmax(axis=1)
    cls = KNeighborsClassifier(n_neighbors=1)
    cls.fit(X_train,Y_train)
    Y = cls.predict(X_test)
    M = len(Y)
    res = (Y == Y_test).sum()
    print(f"Точность предсказания равна {100*res/M}%")

def experiment_indents():
    path = "../../data/MNIST_dataset/mnist_train.csv"

    X_train, Y_train = load_mnist_data(path)

    Y_train = Y_train.argmax(axis=1)
    beg = time.time()
    Gamma = calc_gamma(X_train,Y_train)
    end = time.time()
    print(f"Время вычисления Gamma = {end - beg} сек.")
    beg = time.time()
    indents = calc_indents(Gamma,Y_train)
    end = time.time()
    print(f"Время вычисления отступов = {end - beg} сек.")
    m = (indents < 0).sum()/indents.shape[0]
    print(f"Доля образцов с отрицательным отступом: {m*100}%")


if __name__ == "__main__":
    experiment()
    # test_neibours()
    # experiment_indents()
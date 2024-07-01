import numpy as np
import random


def load_mnist_data(fname):
    f = open(fname)
    X = []
    Y = []
    for line in f:
        data = line.split(",")
        digit = int(data[0])
        x = np.array(data[1:]).astype(np.float32)
        y = np.zeros((10,),dtype=float)
        y[digit] = 1.0
        Y.append(y)
        X.append(x)

    X = np.array(X)/255
    Y = np.array(Y)
    return X,Y


def test_load_data():
    path = "../data/MNIST_dataset/mnist_train_100.csv"
    X,Y = load_mnist_data(path)
    ind = random.randint(0,X.shape[0])
    print(X[ind])
    print(Y[ind])


if __name__ == "__main__":
    test_load_data()

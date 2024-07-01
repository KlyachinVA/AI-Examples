import random

import numpy as np
from scipy import linalg
import dataload as dl
import time
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import cv2 as cv


def fit(X,Y):
    eps = 0.0001
    E = np.eye(X.shape[1])
    X1 = X.T.dot(X)
    Y1 = Y.T.dot(X)
    try:
        XR = linalg.inv(X1)
    except:
        XR = linalg.inv(eps*E + X1)
    A = Y1.dot(XR)
    return A

def test_fit():
    N = 16
    n = 2
    B = np.array([[1,-1],[1,1.0]])

    X = np.random.random((N,n))
    Y = B.dot(X.T).T

    A = fit(X,Y)

    print(A)



def predict(A,X):
    return X.dot(A.T)


def test_predict():
    N = 16
    M = 3
    n = 2
    eps = 0.01
    B = np.array([[1, -1], [1, 1.0]])
    X_train = np.random.random((N, n))
    Y_train = X_train.dot(B.T) #B.dot(X.T).T

    A = fit(X_train, Y_train)

    X_test = np.random.random((M, n))
    Y_test = X_test.dot(B.T) + eps*np.random.random((M,n))
    y = predict(A,X_test)
    e = y - Y_test
    e = (e*e).sum()
    print(e)


def test_mnist_regression():
    path = "../data/MNIST_dataset/mnist_train.npz"
    path_test = "../data/MNIST_dataset/mnist_test.npz"
    begin = time.time()
    print("Загрузка обучающей выборки (60 000) ...")
    X_train, Y_train = dl.load_from_npz(path)

    print(f"Закончено за время {time.time() - begin}. Загружено {X_train.shape[0]} образцов")
    print(f"Размер данных = {X_train.shape}")
    begin = time.time()
    print("Загрузка тестовой выборки (10 000) ...")
    X_test, Y_test = dl.load_from_npz(path_test)
    print(f"Закончено за время {time.time() - begin}. Загружено {X_test.shape[0]} образцов")
    print(f"Размер данных = {X_test.shape}")
    begin = time.time()
    print("Обучение ...")
    A = fit(X_train, Y_train)
    print(f"Закончено за время {time.time() - begin}")
    begin = time.time()
    print("Предсказание ...")
    Y = predict(A, X_test)
    print(f"Закончено за время {time.time() - begin}")
    # print(Y_test.argmax(axis=1))
    # print(Y.argmax(axis=1))

    d1 = Y_test.argmax(axis=1)
    d2 = Y.argmax(axis=1)
    M = Y_test.shape[0]
    d = (d1 == d2).sum()
    print(f"Точность предсказания: {d*100/M}%")

def sigma(x):
    return x + 0.5*x*x - 0.3 * x*x*x
def pca_experiment():
    path = "../data/MNIST_dataset/mnist_train.npz"
    path_test = "../data/MNIST_dataset/mnist_test.npz"
    begin = time.time()
    print("Загрузка обучающей выборки (60 000) ...")
    X_train, Y_train = dl.load_from_npz(path)

    print(f"Закончено за время {time.time() - begin}. Загружено {X_train.shape[0]} образцов")
    print(f"Размер данных = {X_train.shape}")
    begin = time.time()
    print("Загрузка тестовой выборки (10 000) ...")
    X_test, Y_test = dl.load_from_npz(path_test)
    print(f"Закончено за время {time.time() - begin}. Загружено {X_test.shape[0]} образцов")
    print(f"Размер данных = {X_test.shape}")

    pca = PCA(n_components=90)
    X_trainT = pca.fit_transform(X_train)
    X_testT = pca.transform(X_test)
    begin = time.time()
    print("Обучение ...")
    A = fit(X_trainT, Y_train)
    print(f"Закончено за время {time.time() - begin}")
    begin = time.time()
    print("Предсказание ...")
    Y = predict(A, X_testT)
    print(f"Закончено за время {time.time() - begin}")
    # print(Y_test.argmax(axis=1))
    # print(Y.argmax(axis=1))

    d1 = Y_test.argmax(axis=1)

    d2 = Y.argmax(axis=1)
    M = Y_test.shape[0]
    d = (d1 == d2).sum()
    print(f"Точность предсказания: {d * 100 / M}%")




if __name__=="__main__":
    # test_fit()
    # test_predict()
    # test_mnist_regression()
    # pca_experiment()

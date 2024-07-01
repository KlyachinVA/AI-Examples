import random

import numpy as np
from scipy import linalg
import dataload as dl
import time
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import cv2 as cv

def fitQ(X,Y):
    eps = 0.01
    m = Y.shape[1]
    E = np.eye(X.shape[1])
    Em = np.eye(Y.shape[1])
    G = X.T.dot(X)
    ZT = Y.T.dot(X)
    Z = ZT.T
    V = Y.T.dot(Y)

    V1 = linalg.inv(V)# + eps*Em)
    U = V1.dot(ZT)
    W = linalg.inv(G - Z.dot(U) + eps*E)

    lam = -m/(np.trace(U.dot(W.dot(U.T)) - V1))
    A = -lam*(U.dot(W))
    B = A.dot(U.T) - lam*V1
    return A, B


def fitZ(X,Y):
    eps = 0.0001
    N = X.shape[0]
    n = X.shape[1]
    m = Y.shape[1]
    E = np.eye(X.shape[1])
    X1 = X.T.dot(X)
    Y1 = Y.T.dot(X)
    NS = Y.sum(axis=0)
    for k in range(m):
        Y1[k] /= NS[k]
    YZ = np.zeros((N,n))
    for i in range(N):
        ni = np.argmax(Y[i])
        YZ[i] = Y1[ni]

    ZTX = YZ.T.dot(X)


    try:
        XR = linalg.inv(X1)
    except:
        XR = linalg.inv(eps*E + X1)
    W = ZTX.dot(XR)
    V = linalg.eig(W.T)[1]
    A = V.T[:m]
    return A,Y1


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

def test_fitQ():
    N = 16
    n = 2
    G = np.array([[1,-1],[1,1.0]])
    X = np.random.random((N,n))
    Y = G.dot(X.T).T

    A,B = fitQ(X,Y)

    print(A,B)


def predict(A,X):
    return X.dot(A.T)

def predictZ(A,Z,X):
    N = X.shape[0]
    n = A.shape[1]
    m = A.shape[0]

    Z = Z.dot(A.T)

    Y = X.dot(A.T)
    res = np.zeros((N,m),dtype=float)
    for i in range(N):
        DY = Y[i] - Z
        k = np.argmin((DY*DY).sum(axis=1))
        res[i,k] = 1.0
    return res

def predictQ(A,B,Z,X):
    N = X.shape[0]
    n = A.shape[1]
    m = A.shape[0]

    Z = Z.dot(B.T)

    Y = X.dot(A.T)
    res = np.zeros((N,m),dtype=float)
    for i in range(N):
        DY = Y[i] - Z
        k = np.argmin((DY*DY).sum(axis=1))
        res[i,k] = 1.0
    return res


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

def test_mnist_Zregression():
    path = "../data/MNIST_dataset/mnist_train.csv"
    path_test = "../data/MNIST_dataset/mnist_test.csv"
    begin = time.time()
    print("Загрузка обучающей выборки  ...")
    X_train, Y_train = dl.load_mnist_data(path)

    print(f"Закончено за время {time.time() - begin}. Загружено {X_train.shape[0]} образцов")
    print(f"Размер данных = {X_train.shape}")
    begin = time.time()
    print("Загрузка тестовой выборки  ...")
    X_test, Y_test = dl.load_mnist_data(path_test)
    print(f"Закончено за время {time.time() - begin}. Загружено {X_test.shape[0]} образцов")
    print(f"Размер данных = {X_test.shape}")
    begin = time.time()
    print("Обучение ...")
    A, Z = fitZ(X_train, Y_train)
    print(f"Закончено за время {time.time() - begin}")
    begin = time.time()
    print("Предсказание ...")
    Y = predictZ(A, Z,X_test)
    print(f"Закончено за время {time.time() - begin}")
    # print(Y_test.argmax(axis=1))
    # print(Y.argmax(axis=1))

    d1 = Y_test.argmax(axis=1)
    d2 = Y.argmax(axis=1)
    M = Y_test.shape[0]
    d = (d1 == d2).sum()
    print(f"Точность предсказания: {d*100/M}%")

def test_mnist_Qregression():
    path = "../data/MNIST_dataset/mnist_train.csv"
    path_test = "../data/MNIST_dataset/mnist_test.csv"
    begin = time.time()
    print("Загрузка обучающей выборки  ...")
    X_train, Y_train = dl.load_mnist_data(path)

    print(f"Закончено за время {time.time() - begin}. Загружено {X_train.shape[0]} образцов")
    print(f"Размер данных = {X_train.shape}")
    begin = time.time()
    print("Загрузка тестовой выборки  ...")
    X_test, Y_test = dl.load_mnist_data(path_test)
    print(f"Закончено за время {time.time() - begin}. Загружено {X_test.shape[0]} образцов")
    print(f"Размер данных = {X_test.shape}")
    begin = time.time()
    print("Обучение ...")
    A, B = fitQ(X_train, Y_train)
    print(B)
    Z = np.eye(B.shape[0])
    print(f"Закончено за время {time.time() - begin}")
    begin = time.time()
    print("Предсказание ...")
    Y = predictQ(A,B,Z,X_test)
    print(f"Закончено за время {time.time() - begin}")
    # print(Y_test.argmax(axis=1))
    # print(Y.argmax(axis=1))

    d1 = Y_test.argmax(axis=1)
    d2 = Y.argmax(axis=1)
    M = Y_test.shape[0]
    d = (d1 == d2).sum()
    print(f"Точность предсказания: {d*100/M}%")


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
def gen_mnist_regression():
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
    eps = 0.05
    k = 8
    N = len(Y_train)
    for j in range(k):
        K1 = random.randint(0,N)
        K2 = random.randint(K1,K1+300)
        Y_train1 = Y_train[K1:K2]
        X_train1 = X_train[K1:K2]
        # E = np.random.normal(size=(len(Y_train1), 10))
        A = fit(Y_train1, X_train1)
        print(f"Закончено за время {time.time() - begin}")
        begin = time.time()
        print("Предсказание ...")
        M = len(Y_test)

        # eps = 0.01
        # E = np.random.normal(size=(M,10))
        X = predict(A, Y_test)
        X = sigma(X)
        X *= 255

        m = 3
        for i in range(m):
            idx = random.randint(0,M)
            img = X[idx].reshape((28,28))
            cv.imwrite("../data/mnist_gen/img" + str(K1)+str(K2)+str(i) + ".png",img)




def test_cats_dogs_regression():
    m = 16
    path = f"../data/cats_and_dogs_small/cats_dogs_train_{m}.npz"
    path_test = f"../data/cats_and_dogs_small/cats_dogs_test_{m}.npz"
    begin = time.time()
    print("Загрузка обучающей выборки  ...")
    X_train, Y_train = dl.load_from_npz(path)

    print(f"Закончено за время {time.time() - begin}. Загружено {X_train.shape[0]} образцов")
    print(f"Размер данных = {X_train.shape}")
    begin = time.time()
    print("Загрузка тестовой выборки  ...")
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

    Y_test = Y_test.reshape((Y_test.shape[0], 2, m))
    Y_test = (Y_test * Y_test).sum(axis=2)
    # d01 = Y_test[:,:m].mean(axis = 1)
    # print(d01[:10])
    # d11 = Y_test[:,m:2*m].mean(axis=1)
    # print(d11[:10])
    # d1 = 0 + (d11 > d01)
    # print(d1[:10])

    Y = Y.reshape((Y.shape[0], 2, m))
    Y = (Y * Y).sum(axis=2)
    # d02 = Y[:,:m].mean(axis=1)
    # d12 = Y[:,m:2 * m].mean(axis=1)
    # print(d02[:10])
    # print(d12[:10])
    # d2 = 0 + (d12 > d02)
    # print(d2[:10])
    d1 = Y_test.argmax(axis=1)
    d2 = Y.argmax(axis=1)
    M = Y_test.shape[0]
    d = (d1 == d2).sum()
    print(f"Точность предсказания: {d*100/M}%")


def test_mnist_multidim_regression():
    m = 2
    path = "../data/MNIST_dataset/mnist_multidim_train_" + str(m) + ".npz"
    path_test = "../data/MNIST_dataset/mnist_multidim_test_" + str(m) + ".npz"
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
    Y1 = predict(A, X_test)
    print(f"Закончено за время {time.time() - begin}")
    # print(Y_test.argmax(axis=1))
    # print(Y.argmax(axis=1))

    Y_test = Y_test.reshape((Y_test.shape[0],10,m))
    Y_test = (Y_test * Y_test).sum(axis=2)

    Y1 = Y1.reshape((Y1.shape[0], 10, m))
    Y1 = (Y1 * Y1).sum(axis=2)


    d1 = Y_test.argmax(axis=1)
    d2 = Y1.argmax(axis=1)
    M = Y_test.shape[0]
    d = (d1 == d2).sum()
    print(f"Точность предсказания: {d*100/M}%")



def test_velo_var_regression():

    # path = "../data/velo/velo_var_data9.npz"
    path = "../data/velo/velo_var_dim1_data9.npz"
    # path = "../data/velo/velo_data4.npz"
    # d = np.load("../data/velo/B.npz")
    # B = d["B"]
    # lam = 0.0007

    begin = time.time()
    print("Загрузка набора данных  ...")
    X, Y = dl.load_from_npz(path)


    print(f"Закончено за время {time.time() - begin}. Загружено {X.shape[0]} образцов")
    print(f"Размер данных = {X.shape}")
    begin = time.time()
    N = X.shape[0]
    X_train = X[:2*N//3]
    Y_train = Y[:2*N//3]
    X_test = X[2*N//3:]
    Y_test = Y[2*N//3:]
    begin = time.time()
    print("Обучение ...")
    A = fit(X_train, Y_train)
    print(f"Закончено за время {time.time() - begin}")
    begin = time.time()
    print("Предсказание ...")
    # A = (A - lam * B)/(1-lam)
    Y1 = predict(A, X_test)
    print(f"Закончено за время {time.time() - begin}")
    # print(Y_test.argmax(axis=1))
    # print(Y.argmax(axis=1))



    # d1 = Y_test.argmax(axis=1)
    # d2 = Y1.argmax(axis=1)
    d1 = Y_test>0
    d2 = Y1 > 0
    print(f"d1==1 = {(d1==1).sum()/len(d1)}")
    print(f"d2==1 = {(d2 == 1).sum() / len(d2)}")
    M = Y_test.shape[0]
    d = (d1 == d2).sum()
    print(f"Точность предсказания: {d*100/M}%")

def test_velo3_var_regression():

    path = "../data/velo/velo3_var_data1.npz"
    # path = "../data/velo/velo_data4.npz"
    d = np.load("../data/velo/B.npz")
    B = d["B"]
    lam = 1.0

    begin = time.time()
    print("Загрузка набора данных  ...")
    data = np.load(path)
    X = data["X"]
    Y = data["Y"]
    Y0 = data["Y0"]


    print(f"Закончено за время {time.time() - begin}. Загружено {X.shape[0]} образцов")
    print(f"Размер данных = {X.shape}")
    begin = time.time()
    N = X.shape[0]
    X_train = X[:2*N//3]
    Y_train = Y[:2*N//3]
    X_test = X[2*N//3:]
    Y_test = Y[2*N//3:]
    Y0_test = Y0[2*N//3:]
    Y0_train = Y0[:2*N//3]
    begin = time.time()
    print("Обучение ...")
    A = fit(X_train, Y_train)
    print(f"Закончено за время {time.time() - begin}")
    begin = time.time()
    print("Предсказание ...")
    A = (A - lam * B)
    # Y1 = predict(A, X_test)
    Y1 = predict(A, X_train)
    print(f"Закончено за время {time.time() - begin}")
    # print(Y_test.argmax(axis=1))
    # print(Y.argmax(axis=1))


    print(Y1.shape)
    # d1 = Y0_test.argmax(axis=1)
    d1 = Y_train.argmax(axis=1)
    d2 = Y1.argmax(axis=1)
    print(d1.shape)
    print(d2.shape)
    print(f"d1==1 = {(d1==1).sum()/len(d1)}")
    print(f"d2==1 = {(d2 == 1).sum() / len(d2)}")
    M = d2.shape[0]
    d = (d1 == d2).sum()
    print(f"Точность предсказания: {d*100/M}%")


def test_mnist_var_regression():
    need_save = True
    path = "../data/MNIST_dataset/mnist_var_train" + ".npz"
    path_test = "../data/MNIST_dataset/mnist_var_test" + ".npz"
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
    if need_save:
        D = {"B" : A}
        np.savez_compressed("../data/MNIST_dataset/B.npz",**D)
    begin = time.time()
    print("Предсказание ...")
    Y1 = predict(A, X_test)
    print(f"Закончено за время {time.time() - begin}")
    # print(Y_test.argmax(axis=1))
    # print(Y.argmax(axis=1))



    d1 = Y_test.argmax(axis=1)
    for k in range(10):
        print(f"d1=={k} = {(d1==k).sum()/len(d1)}")
    d2 = Y1.argmax(axis=1)
    M = Y_test.shape[0]
    d = (d1 == d2).sum()
    print(f"Точность предсказания: {d*100/M}%")

def trans_to_digit(Y):
    d = np.zeros((Y.shape[0],),dtype=int)
    for i in range(Y.shape[0]):
        y = Y[i]
        w = np.zeros((10,),dtype=float)
        w[::2] = -y
        w[1::2] = y
        d[i] = w.argmin()

    return d

def test_mnist_var5_regression():
    need_save = True
    path = "../data/MNIST_dataset/mnist_var5_train" + ".npz"
    path_test = "../data/MNIST_dataset/mnist_var5_test" + ".npz"
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
    if need_save:
        D = {"B" : A}
        np.savez_compressed("../data/MNIST_dataset/B.npz",**D)
    begin = time.time()
    print("Предсказание ...")
    Y1 = predict(A, X_test)
    print(f"Закончено за время {time.time() - begin}")
    # print(Y_test.argmax(axis=1))
    # print(Y.argmax(axis=1))


    # threshold = 0.999995
    # Y_test[Y_test>threshold] = 1
    # Y_test[Y_test<=-threshold] = -1
    # Y_test[np.logical_and(Y_test>-threshold,Y_test<=threshold)] =0
    #
    # Y1[Y1 > threshold] = 1
    # Y1[Y1 <= -threshold] = -1
    # Y1[np.logical_and(Y1 > -threshold, Y1 <= threshold)] = 0
    d1 = trans_to_digit(Y_test)
    d2 = trans_to_digit(Y1)
    for k in range(10):
        print(f"=={k} = {(d1==k).sum()/len(d1)}")

    M = Y_test.shape[0]
    fr = (d1 == d2).sum()
    print(f"Точность предсказания: {fr*100/M}%")

def test_velo_multidim_regression():
    m = 3
    path = "../data/velo/velo_m_" + str(m) + "_data.npz"
    begin = time.time()
    print("Загрузка данных ...")
    X, Y = dl.load_from_npz(path)

    print(f"Закончено за время {time.time() - begin}. Загружено {X.shape[0]} образцов")
    print(f"Размер данных = {X.shape}")
    begin = time.time()
    N = X.shape[0]

    X_train = X[:2*N//3]
    Y_train = Y[:2*N//3]
    X_test = X[2 * N // 3:]
    Y_test = Y[2 * N // 3:]

    begin = time.time()
    print("Обучение ...")
    A = fit(X_train, Y_train)
    print(f"Закончено за время {time.time() - begin}")
    begin = time.time()
    print("Предсказание ...")
    Y1 = predict(A, X_test)
    print(Y1.shape)
    print(f"Закончено за время {time.time() - begin}")
    # print(Y_test.argmax(axis=1))
    # print(Y.argmax(axis=1))

    Y_test = Y_test.reshape((Y_test.shape[0],2,m))
    Y_test = (Y_test * Y_test).sum(axis=2)

    Y1 = Y1.reshape((Y1.shape[0], 2, m))
    Y1 = (Y1 * Y1).sum(axis=2)


    d1 = Y_test.argmax(axis=1)
    d2 = Y1.argmax(axis=1)
    print(d1[-10:])
    print(d2[-10:])
    M = Y_test.shape[0]
    d = (d1 == d2).sum()
    print(f"Точность предсказания: {d*100/M}%")



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


def test_velo_regression():
    path = "../data/velo/velo_var_data9.npz"

    begin = time.time()
    print("Загрузка данных ...")
    X, Y = dl.load_from_npz(path)

    print(f"Закончено за время {time.time() - begin}. Загружено {X.shape[0]} образцов")
    print(f"Размер данных = {X.shape}")
    N = X.shape[0]
    X_train = X[:2*N//3]
    Y_train = Y[:2*N//3]
    X_test = X[2*N//3:]
    Y_test = Y[2*N//3:]
    begin = time.time()
    print("Обучение ...")
    A = fit(X_train, Y_train)
    print(f"Закончено за время {time.time() - begin}")
    begin = time.time()
    print("Предсказание ...")
    Y1 = predict(A, X_test)
    print(f"Закончено за время {time.time() - begin}")
    # print(Y_test.argmax(axis=1))
    # print(Y.argmax(axis=1))

    d1 = Y_test.argmax(axis=1)
    d2 = Y1.argmax(axis=1)
    M = Y_test.shape[0]
    d = (d1 == d2).sum()
    print(f"Точность предсказания: {d*100/M}%")


if __name__=="__main__":
    # test_fit()
    # test_predict()
    # test_mnist_regression()
    gen_mnist_regression()
    # test_mnist_Zregression()
    # test_mnist_multidim_regression()
    # test_mnist_var_regression()
    # test_mnist_var5_regression()
    # pca_experiment()
    # test_cats_dogs_regression()
    # test_fitQ()
    # test_mnist_Qregression()

    # test_velo_regression()
    # test_velo_multidim_regression()
    # test_velo_var_regression()
    # test_velo3_var_regression()
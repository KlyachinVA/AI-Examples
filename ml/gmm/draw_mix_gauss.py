import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.mixture import GaussianMixture as GMM

def x(t):
    return t**2 +t**4 - 10

def y(t):
    return t**3 - t


def add_noise(x,y,a,b,N,eps):
    X = []
    Y = []
    for i in range(N):
        t = a + i*(b-a)/N
        x0 = x(t)
        y0 = y(t)
        x0 += eps*(random.random()-0.5)
        y0 += eps * (random.random() - 0.5)
        X.append(x0)
        Y.append(y0)

    return X,Y

def approx_gauss(x,y,a,b,N,eps):
    X,Y = add_noise(x,y,a,b,N,eps)
    n = 70
    PX = np.zeros((N*n,),dtype=float)
    PY = np.zeros((N * n,), dtype=float)
    print(PX.shape)
    i = 0
    for x0,y0 in zip(X,Y):
        QX = np.random.normal(loc=x0,scale=eps,size=n)
        QY = np.random.normal(loc=y0, scale=eps, size=n)
        print(QX.shape)
        PX[i*n:n*(i+1)] = QX
        PY[i*n:n*(i+1)] = QY
        i+=1
    plt.plot(PX,PY,'.')
    plt.grid(True)
    plt.show()




def run():
    a = -1
    b = 1
    N = 300
    eps = 0.1
    X,Y = add_noise(x,y,a,b,N,eps)

    plt.plot(X,Y,'.')
    plt.grid(True)
    plt.show()

def go():
    a = -1
    b = 1
    N = 30
    eps = 0.05
    approx_gauss(x, y, a, b, N, eps)


def use_gmm():
    a = -1
    b = 1
    N = 700
    eps = 0.1
    X, Y = add_noise(x, y, a, b, N, eps)

    plt.plot(X, Y, '.')
    plt.grid(True)
    plt.show()

    gmm = GMM(n_components=8, covariance_type='full')
    data = np.zeros((N,2),dtype=float)
    data[:,0] = X
    data[:,1] = Y
    gmm.fit(data)
    print(gmm.converged_)

    data_new = gmm.sample(500)  # , random_state=0)
    print(data_new[0].shape)
    plt.plot(data_new[0][:,0],data_new[0][:,1],'.')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # run()
    # go()
    use_gmm()


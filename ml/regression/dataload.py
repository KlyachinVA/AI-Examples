import numpy as np
import random
import math
import os
import cv2 as cv


def load_mnist_multi_data(fname):
    f = open(fname)
    X = [[],[],[],[],[],[],[],[],[],[]]
    Y = [[],[],[],[],[],[],[],[],[],[]]
    for line in f:
        data = line.split(",")
        data.append("255")
        digit = int(data[0])
        x = np.array(data[1:]).astype(np.float32)
        y = np.zeros((10,),dtype=float)
        y[digit] = 1.0
        Y[digit].append(y)
        X[digit].append(x)
    for i in range(10):
        X[i] = np.array(X[i])/255
        Y[i] = np.array(Y[i])


    return X,Y


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

def load_mnist_inhomogeneous_data(fname):
    f = open(fname)
    X = []
    Y = []
    for line in f:
        data = line.split(",")
        digit = int(data[0])
        d = data[1:]
        d.append("255")
        # print(d)
        x = np.array(d).astype(np.float32)
        y = np.zeros((10,),dtype=float)
        y[digit] = 1.0
        Y.append(y)
        X.append(x)

    X = np.array(X)/255
    Y = np.array(Y)
    return X,Y


def create_npz(X,Y,fname):
    data = {"X":X,"Y":Y}
    np.savez_compressed(fname,**data)

def multi_mnist_to_npz(fname,out_fname):
    X,Y = load_mnist_multi_data(fname)
    for i in range(10):
        create_npz(X[i],Y[i],out_fname[:-4]+str(i)+".npz")

def load_from_npz(fname):
    f = np.load(fname)
    return f["X"],f["Y"]
def load_multidim_mnist_data(fname,m):
    f = open(fname)
    X = []
    Y = []

    ei = np.eye(m)


    for line in f:
        data = line.split(",")
        data.append("255")
        digit = int(data[0])
        x = np.array(data[1:]).astype(np.float32)
        y = np.zeros((10*m,),dtype=float)
        # for k in range(m):
        e = np.random.random((m,))
        L = math.sqrt((e*e).sum())
        e /= L
        eps = 0.003
        z = e * (1.6+eps*np.random.random(1))
        r = 0.2
        # z = ei[0]
        # z += r*ei[m-1] + 0.0 * np.random.normal(loc=0.001,size=(m,)) #np.random.random((m,))
        # z += r*e
            # z = 1 + 0.001*np.random.random((m,))
            #z = 1.6*np.ones((m,),dtype=float) + 0.05 * np.random.normal((m,))
            # z = (2.4+5.0*random.random())*np.ones((m,),dtype=float)
            #phi = random.random()*np.pi

            # z += (a*ei[0]*math.cos(phi) + b*ei[1]*math.sin(phi) +\
            #       a*ei[2]*math.sin(phi) - b*ei[3]*math.cos(phi)) +\
            #       a*ei[4]*phi
        y[digit*m:digit*m+m] = z

        Y.append(y)
        X.append(x)

    X = np.array(X)/255
    Y = np.array(Y)
    return X,Y

def load_var5_mnist_data(fname,B):
    f = open(fname)
    X = []
    Y = []


    lam = 0.000
    for line in f:
        # B = np.random.random((10,785))
        data = line.split(",")
        data.append("255")
        digit = int(data[0])
        x = np.array(data[1:]).astype(np.float32)/255.0
        y = np.zeros((5,),dtype=float)
        # for k in range(m):
        if digit%2 == 0:
            y[digit//2] = 5.0
        else:
            y[digit//2] = -5.0
        y = y + lam*(B.dot(x) - y)

        Y.append(y)
        X.append(x)

    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def load_var_mnist_data(fname,B):
    f = open(fname)
    X = []
    Y = []


    lam = -4.7
    for line in f:
        # B = np.random.random((10,785))
        data = line.split(",")
        data.append("255")
        digit = int(data[0])
        x = np.array(data[1:]).astype(np.float32)/255.0
        y = np.zeros((10,),dtype=float)
        # for k in range(m):
        y[digit] = 1.0
        y = y + lam*(B.dot(x) - y)

        Y.append(y)
        X.append(x)

    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def create_velo_npz(fname,out_fname):
    f = open(fname)

    XY = []
    maxx = 0
    for line in f:
        data = line.split(";")
        label = int(data[-1])
        xy = np.zeros(len(data)+1,dtype=float)
        xy[:-2] = np.array(data[:-1]).astype(np.float32)
        # maxx = xy[:-2].max()
        # xy[:-2] /= maxx
        y = np.zeros((2,),dtype=float)
        y[label] = 1.0
        xy[-2:] = y

        XY.append(xy)
    random.shuffle(XY)
    random.shuffle(XY)
    XY = np.array(XY)
    X = np.array(XY[:,:-2])
    Y = np.array(XY[:,-2:])
    D = {"X":X,"Y":Y}
    np.savez_compressed(out_fname,**D)


def create_npz_var_velo_data(fname,out_fname):
    f = open(fname)
    XY = []
    # n = 275
    n = 853
    maxx = 0
    k = 1.0
    needB = False
    if needB:
        B = k * np.random.random((1,n))

        DB = {"B":B}
        np.savez_compressed("../data/velo/B.npz", **DB)
    else:
        d = np.load("../data/velo/B.npz")
        B = d["B"]

    # lam = 0.0016
    lam = 0.0
    # lam = 0.0008
    for line in f:
        data = line.split(";")
        # B = k * np.random.random((2, n))
        # B = k * (np.random.random((1, n)) - 0.0)

        label = int(data[-1])
        # xy = np.zeros(len(data) + 1,dtype=float)
        xy = np.zeros(len(data) , dtype=float)
        # xy[:-2] = np.array(data[:-1]).astype(np.float32)
        xy[:-1] = np.array(data[:-1]).astype(np.float32)
        # maxx = xy[:-2].max()
        xy[:-2] /= 500
        # y = np.zeros((2,),dtype=float)
        y = np.zeros((1,), dtype=float)

        # y[label] = 1.0
        y[0] = 2*label - 1.0
        # y = (1 - lam)*y + lam*(B.dot(xy[:-2]) - y)
        # xy[-2:] = y

        y = (1 - lam) * y + lam * (B.dot(xy[:-1]) - y)
        xy[-1:] = y


        XY.append(xy)
    random.shuffle(XY)
    random.shuffle(XY)
    XY = np.array(XY)
    # X = XY[:,:-2]
    # Y = XY[:,-2:]
    X = XY[:, :-1]
    Y = XY[:, -1:]
    print(Y)

    D = {"X": X, "Y": Y}
    np.savez_compressed(out_fname, **D)


def create_npz_var_velo3_data(fname,out_fname):
    f = open(fname)
    XY = []
    # n = 275
    n = 853
    maxx = 0
    k = 0.9
    needB = True
    if needB:
        B = k * np.random.random((2,n))

        DB = {"B":B}
        np.savez_compressed("../data/velo/B.npz", **DB)
    else:
        d = np.load("../data/velo/B.npz")
        B = d["B"]

    lam = 1.0
    for line in f:
        data = line.split(";")
        # B = k * np.random.random((2, n))
        label = int(data[-1])
        xy = np.zeros(len(data) + 3,dtype=float)
        xy[:-4] = np.array(data[:-1]).astype(np.float32)
        maxx = xy[:-4].max()
        xy[:-4] /= maxx
        y = np.zeros((4,),dtype=float)

        y[label] = 1.0
        y[label + 2] =1.0
        y[:2] = (1 - lam)*y[:2] + lam*(B.dot(xy[:-4]) - y[:2])
        xy[-4:] = y


        XY.append(xy)
    random.shuffle(XY)
    random.shuffle(XY)
    XY = np.array(XY)
    X = XY[:,:-4]
    Y = XY[:,-4:-2]
    Y0 = XY[:,-2:]
    print(Y[-10:])
    print(Y0[-10:])

    D = {"X": X, "Y": Y,"Y0":Y0}
    np.savez_compressed(out_fname, **D)



def create_npz_multidim_velo_data(fname,m,out_fname):
    f = open(fname)
    XY = []


    ei = np.eye(m)

    a = 0.08 + 0.009*random.random()
    b = 0.0012 + 0.007*random.random()
    for line in f:
        data = line.split(";")

        label = int(data[-1])

        xy = np.zeros(len(data)+2*m -1,dtype=float)
        xy[:-2*m] = np.array(data[:-1]).astype(np.float32)
        y = np.zeros((2*m,),dtype=float)
        # for k in range(m):
        e = np.random.random((m,))
        L = math.sqrt((e*e).sum())
        e /= L
        r = 0.23
        z = ei[0]*0.1
        z += 1.1*ei[m-1] + 0.001 * np.random.random((m,))#np.random.normal(loc=0.01,size=(m,)) #np.random.random((m,))
        z += r*e
            # z = 1 + 0.001*np.random.random((m,))
            #z = 1.6*np.ones((m,),dtype=float) + 0.05 * np.random.normal((m,))
            # z = (2.4+5.0*random.random())*np.ones((m,),dtype=float)
            #phi = random.random()*np.pi

            # z += (a*ei[0]*math.cos(phi) + b*ei[1]*math.sin(phi) +\
            #       a*ei[2]*math.sin(phi) - b*ei[3]*math.cos(phi)) +\
            #       a*ei[4]*phi
        y[label*m:label*m+m] = z
        xy[-2*m:] = y
        print(label,y)


        XY.append(xy)
    random.shuffle(XY)
    XY = np.array(XY)
    X = XY[:,:-2*m]
    Y = XY[:,-2*m:]
    D = {"X":X,"Y":Y}
    np.savez_compressed(out_fname,**D)


def create_multidim_npz():
    m = 2
    # type_data = "train"
    type_data = "test"
    path_out = "../data/MNIST_dataset/mnist_multidim_" + type_data + "_" + str(m) + ".npz"
    path = "../data/MNIST_dataset/mnist_" + type_data + ".csv"

    X, Y = load_multidim_mnist_data(path, m)
    create_npz(X, Y, path_out)

    type_data = "train"
    path_out = "../data/MNIST_dataset/mnist_multidim_" + type_data + "_" + str(m) + ".npz"
    path = "../data/MNIST_dataset/mnist_" + type_data + ".csv"

    X, Y = load_multidim_mnist_data(path, m)
    create_npz(X, Y, path_out)

def create_var5_npz():
    n = 785
    from_file = False
    if from_file:
        data = np.load("../data/MNIST_dataset/B.npz")
        B = data["B"]

    else:
        B = 1.0 * (np.random.random((5, n)) )

    # B = np.zeros((10,n),dtype=float)
    # type_data = "train"
    type_data = "test"
    path_out = "../data/MNIST_dataset/mnist_var5_" + type_data + ".npz"
    path = "../data/MNIST_dataset/mnist_" + type_data + ".csv"

    X, Y = load_var5_mnist_data(path,B)
    create_npz(X, Y, path_out)

    type_data = "train"
    path_out = "../data/MNIST_dataset/mnist_var5_" + type_data + ".npz"
    path = "../data/MNIST_dataset/mnist_" + type_data + ".csv"

    X, Y = load_var5_mnist_data(path, B)
    create_npz(X, Y, path_out)

def create_var_npz():
    n = 785
    from_file = False
    if from_file:
        data = np.load("../data/MNIST_dataset/B.npz")
        B = data["B"]

    else:
        B = 0.1 * (np.random.random((10, n)) + 28.0)

    # B = np.zeros((10,n),dtype=float)
    # type_data = "train"
    type_data = "test"
    path_out = "../data/MNIST_dataset/mnist_var_" + type_data + ".npz"
    path = "../data/MNIST_dataset/mnist_" + type_data + ".csv"

    X, Y = load_var_mnist_data(path,B)
    create_npz(X, Y, path_out)

    type_data = "train"
    path_out = "../data/MNIST_dataset/mnist_var_" + type_data + ".npz"
    path = "../data/MNIST_dataset/mnist_" + type_data + ".csv"

    X, Y = load_var_mnist_data(path, B)
    create_npz(X, Y, path_out)


def load_cats_dogs_data(path,size,colored=False,m=1):
    X = []
    Y = []

    path_cats = path + "cats/"
    path_dogs = path + "dogs/"
    ei=np.eye(m)
    for fname in os.listdir(path_cats):
        print(fname)
        img = cv.imread(path_cats + fname)
        img_y = None
        e = np.random.random((m,))
        L = math.sqrt((e * e).sum())
        e /= L
        r = 0.01
        z = ei[0] + 0.01 * np.random.random((m,))
        z += r * e
        if not colored:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img =  cv.resize(img,size)
            img = img.reshape((img.shape[0] * img.shape[1],))
            img_y = np.zeros((img.shape[0]+2*m,),dtype=float)
            img_y[:-2*m] = img/255.0
            img_y[-2*m:-m] = z#1.0


        else:
            img = cv.resize(img, size[:2])
            img = img.reshape((img.shape[0] * img.shape[1] * 3,))

            img_y = np.zeros((img.shape[0] + 2 * m,), dtype=float)
            img_y[:-2 * m] = img / 255.0
            img_y[-2 * m:-m] = z#1.0
        X.append(img_y)
        # print(img.shape)


    for fname in os.listdir(path_dogs):
        print(fname)
        img = cv.imread(path_dogs + fname)
        img_y = None
        e = np.random.random((m,))
        L = math.sqrt((e * e).sum())
        e /= L
        r = 0.01
        z = ei[1] + 0.01 * np.random.random((m,))
        z += r * e
        if not colored:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img =  cv.resize(img,size)
            img = img.reshape((img.shape[0] * img.shape[1],))
            img_y = np.zeros((img.shape[0] + 2 * m,), dtype=float)
            img_y[:-2 * m] = img / 255.0
            img_y[-m:] = z#1.0
        else:
            img = cv.resize(img, size[:2])
            img = img.reshape((img.shape[0] * img.shape[1] * 3,))
            img_y = np.zeros((img.shape[0] + 2 * m,), dtype=float)
            img_y[:-2 * m] = img / 255.0
            img_y[-m:] = z#1.0
        # print(img.shape)
        X.append(img_y)

    random.shuffle(X)
    X = np.array(X)
    print(X.shape)
    X1 = X[:,:-2*m]
    Y1 = X[:,-2*m:]
    print(X1.shape,Y1.shape)
    return X1,Y1



def test_load_data():
    path = "../data/MNIST_dataset/mnist_train_100.csv"
    X,Y = load_mnist_data(path)
    ind = random.randint(0,X.shape[0])
    print(X[ind])
    print(Y[ind])

def test_load_multidim_data():
    path = "../data/MNIST_dataset/mnist_train_100.csv"
    X,Y = load_multidim_mnist_data(path,2)
    ind = random.randint(0,X.shape[0])
    print(X[ind])
    print(Y[ind])

def test_multi_to_npz():
    fname = "../data/MNIST_dataset/mnist_test.csv"
    out_fname = "../data/MNIST_dataset/mnist_multi_test.npz"
    multi_mnist_to_npz(fname,out_fname)

def test_create_inhom_npz():
    fname = "../data/MNIST_dataset/mnist_test_inhom.npz"
    path = "../data/MNIST_dataset/mnist_test.csv"
    X, Y = load_mnist_inhomogeneous_data(path)
    # X, Y = load_mnist_data(path)
    print(X.shape,Y.shape)
    create_npz(X,Y,fname)


def test_create_cat_dog_npz():
    # fname = "../data/MNIST_dataset/mnist_test.npz"
    # path = "../data/MNIST_dataset/mnist_test.csv"
    colored = False
    m = 16
    type_data = "test"
    size = (36, 36)
    if colored:
        size = tuple(list(size) + [3])
    fname = "../data/cats_and_dogs_small/cats_dogs_" + type_data + "_" + str(m) + ".npz"
    path = f"../data/cats_and_dogs_small/{type_data}/"
    X, Y = load_cats_dogs_data(path, size, colored,m)
    # X, Y = load_mnist_data(path)
    create_npz(X,Y,fname)

def test_load_from_npz():
    fname = "../data/MNIST_dataset/mnist_train.npz"
    X,Y = load_from_npz(fname)
    print(X.shape,Y.shape)

def test_load_cats_dogs():
    colored = True
    m = 2
    size = (36, 36)
    if colored:
        size = tuple(list(size) + [3])

    path = "../data/cats_and_dogs_small/test/"
    X, Y = load_cats_dogs_data(path,size,colored,m)
    print(X.shape,Y.shape)

def show_data():
    fname = "../data/MNIST_dataset/mnist_train.npz"
    X, Y = load_from_npz(fname)
    print(X.shape, Y.shape)
    k = 47
    im = X[k].reshape((28,28))
    print(np.argmax(Y[k]))
    cv.imshow("MNIST",im)
    cv.waitKey()

if __name__ == "__main__":
    # test_load_data()
    # test_load_multidim_data()
    # test_create_cat_dog_npz()
    # test_load_from_npz()
    # test_load_cats_dogs()
    # show_data()
    create_multidim_npz()
    # create_var_npz()
    # create_var5_npz()
    # test_multi_to_npz()
    # test_create_inhom_npz()

    # create_velo_npz("../data/velo/data_target_velo-265.txt",
    #                 "../data/velo/velo_data4.npz")

    # create_velo_npz("../data/velo/data_target_velo-4000-d.txt",
    #                 "../data/velo/velo_data4.npz")

    # m = 3
    # create_npz_multidim_velo_data("../data/velo/data_target_velo-265.txt",
    #                               m,
    #                               f"../data/velo/velo_m_{m}_data.npz")
    # create_npz_var_velo_data("../data/velo/data_target_velo-265.txt",
    #                               f"../data/velo/velo_var_data9.npz")
    # create_npz_var_velo_data("../data/velo/data_target_velo-4000-d.txt",
    #                          f"../data/velo/velo_var_dim1_data9.npz")

    # create_npz_var_velo3_data("../data/velo/data_target_velo-4000-d.txt",
    #                          f"../data/velo/velo3_var_data1.npz")
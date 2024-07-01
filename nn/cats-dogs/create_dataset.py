import cv2 as cv
import os
import numpy as np
import random
from config import *

def create(path,fname_out,size):

    cats_path = path + "cats/"
    dogs_path = path + "dogs/"

    X = []

    for fname in os.listdir(cats_path):
        print(fname)
        img = cv.imread(cats_path + fname)
        img = cv.resize(img,size)
        img = np.reshape(img,size[0]*size[1]*3)
        x = np.zeros(size[0]*size[1]*3 + 1,dtype=np.uint8)
        x[:-1] = img
        x[-1] = 0
        X.append(x)

    for fname in os.listdir(dogs_path):
        print(fname)
        img = cv.imread(dogs_path + fname)
        img = cv.resize(img,size)
        img = np.reshape(img,size[0]*size[1]*3)
        x = np.zeros(size[0]*size[1]*3 + 1,dtype=np.uint8)
        x[:-1] = img
        x[-1] = 1
        X.append(x)

    random.shuffle(X)
    X = np.array(X)
    Y = X[:,-1]
    X = X[:,:-1]
    data = {"X":X,"Y":Y}
    np.savez_compressed(fname_out,**data)

def test_create():
    create(TEST_PATH,TEST_FNAME,SIZE)
    create(TRAIN_PATH,TRAIN_FNAME,SIZE)

def visual_test():
    data = np.load(TEST_FNAME)
    X = data["X"]
    Y = data["Y"]
    N = len(Y)
    ind = random.randint(0,N-1)
    img = X[ind]
    label = Y[ind]
    img = np.reshape(img,(SIZE[0],SIZE[1],3))
    print(label)
    cv.imshow("Visual test",img)
    cv.waitKey()


if __name__ == "__main__":
    test_create()
    # visual_test()

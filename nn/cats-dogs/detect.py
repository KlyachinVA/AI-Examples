import cv2 as cv
import os
import random
import pickle
import numpy as np
from config import *

def detect(img_fname,fname_model):
    labels = ["кошка","собака"]
    model = pickle.load(open(fname_model,"rb"))

    img = cv.imread(img_fname)
    img = cv.resize(img,SIZE)
    img = img / 255.0
    y = model.predict(np.array([img]))
    print(y,labels[np.argmax(y[0])])

def test_detect():
    val_path = PATH + "validation/all/"

    files = os.listdir(val_path)
    N = len(files)

    ind = random.randint(0,N-1)
    img_fname = files[ind]
    detect(val_path + img_fname, FNAME_MODEL)
    cv.imshow("Detect image",cv.imread(val_path + img_fname))
    cv.waitKey()

if __name__ == "__main__":
    test_detect()
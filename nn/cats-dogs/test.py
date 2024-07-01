import pickle
import numpy as np

from config import *

def test_model():

    fname_data = TEST_FNAME

    model = pickle.load(open(FNAME_MODEL,"rb"))
    data = np.load(fname_data)
    X = data["X"]
    Y = data["Y"]
    X = X / 255.0
    X = np.reshape(X, (len(X), INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
    Y_pred = model.predict(X)
    Y_pred = Y_pred.argmax(axis=1)
    res = (Y==Y_pred).sum()/len(Y)
    print(f"Точность предсказания = {res*100}%")

if __name__ == "__main__":
    test_model()

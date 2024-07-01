import pickle
from config import *
import numpy as np

from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical

def create_model(filters, kernel_size,input_shape,layers,units,num_output):
    model = Sequential()
    model.add(Conv2D(filters=filters,kernel_size=kernel_size,input_shape=input_shape,activation="relu"))

    for layer in layers:
        model.add(Conv2D(filters=layer["filters"],kernel_size=layer["kernel_size"],activation="relu"))

        if layer["maxpool"]:
            model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(units,activation="relu"))
    model.add(Dense(num_output,activation="softmax"))

    return model

def build_model():
    layers =[{"filters":64,"kernel_size":(3,3),"maxpool":True},
             {"filters":32,"kernel_size":(3,3),"maxpool":True}]

    model = create_model(FILTERS,KERNEL_SIZE,INPUT_SHAPE,layers,UNITS,NUM_OUTPUT)
    return model

def test_build():
    model = build_model()
    print(model.summary())

def train(train_fname):

    model = build_model()
    data = np.load(train_fname)
    X = data["X"]
    Y = data["Y"]
    Y = to_categorical(Y,2)
    X = X/255.0
    X = np.reshape(X,(len(X),INPUT_SHAPE[0],INPUT_SHAPE[1],3))
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    model.fit(X,Y,batch_size=BATCH_SIZE,epochs=EPOCHS)
    pickle.dump(model,open(FNAME_MODEL,"wb"))

if __name__ == "__main__":
    train(TRAIN_FNAME)


if __name__ == "__main__":
    test_build()
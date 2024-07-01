from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import dataload as dl
import joblib


def experiment():
    path = "../data/MNIST_dataset/mnist_train.csv"
    path_test = "../data/MNIST_dataset/mnist_test.csv"

    X_train, Y_train = dl.load_mnist_data(path)
    Y_train = Y_train.argmax(axis=1)
    X_test,Y_test = dl.load_mnist_data(path_test)
    Y_test = Y_test.argmax(axis=1)

    # model = GaussianNB()
    # model = MultinomialNB()
    model = BernoulliNB()
    model.fit(X_train,Y_train)
    joblib.dump(model,"bayes_mnist_model.bin")
    ld_model = joblib.load("bayes_mnist_model.bin")
    Y = ld_model.predict(X_test)


    S = (Y == Y_test).sum()
    M = Y.shape[0]
    print(f"Точность = {100*S/M}%")

def experiment_cats_dogs():
    path = "../data/cats_and_dogs_small/cats_dogs_train.npz"
    path_test = "../data/cats_and_dogs_small/cats_dogs_test.npz"

    X_train, Y_train = dl.load_from_npz(path)
    Y_train = Y_train.argmax(axis=1)
    X_test,Y_test = dl.load_from_npz(path_test)
    Y_test = Y_test.argmax(axis=1)

    model = GaussianNB()
    # model = MultinomialNB()
    # model = BernoulliNB()
    model.fit(X_train,Y_train)
    joblib.dump(model,"bayes_cats_dogs_model.bin")
    ld_model = joblib.load("bayes_cats_dogs_model.bin")
    Y = ld_model.predict(X_test)


    S = (Y == Y_test).sum()
    M = Y.shape[0]
    print(f"Точность = {100*S/M}%")


if __name__ == "__main__":
    # experiment()
    experiment_cats_dogs()
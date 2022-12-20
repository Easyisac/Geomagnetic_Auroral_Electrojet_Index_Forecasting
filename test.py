from modelBuilder import *
from model import *
from dataPreparation import *
from dask_ml.model_selection import train_test_split


def runTest1():
    X0, Y = create_value_dataset(loockback=10, loockforward=1)
    X1 = create_image_dataset(loockback=10, loockforward=1)
    X0train, X0test, X1train, X1test, Ytrain, Ytest = train_test_split(X0, X1, Y, test_size=0.1, random_state=42,
                                                                       shuffle=True)
    X0train, X0val, X1train, X1val, Ytrain, Yval = train_test_split(X0train, X1train, Ytrain, test_size=0.01,
                                                                    random_state=42, shuffle=True)
    train_gen = DataGen(X0train, X1train, Ytrain, 48)
    val_gen = DataGen(X0val, X1val, Yval, 48)
    test_gen = DataGen(X0test, X1test, Ytest, 48)

    name = 'mixed_test'
    dir = './results/' + name
    model = mixedInputTest((X0, X1), Y)
    model, history = train_model(model, train_gen, val_gen, dir, name)

if __name__ == '__main__':
    runTest1()

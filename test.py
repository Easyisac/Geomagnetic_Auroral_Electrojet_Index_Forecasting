from modelBuilder import *
from model import *
from dataPreparation import *
from dask_ml.model_selection import train_test_split

def genTest():
    X0, X1, Y, train_gen, val_gen, test_gen = prepareData(lookback=10, lookforward=1, batch_size=1)

    X, Y = train_gen[0]
    print(X[1].shape)


def runTest1():
    X0, X1, Y, train_gen, val_gen, test_gen = prepareData(lookback=1, lookforward=1, batch_size=1)

    name = 'mixed_test'
    dir = './results/' + name
    model = mixedInputTest((X0, X1), Y)
    model, history = train_model(model, train_gen, val_gen, dir, name)

if __name__ == '__main__':
    #genTest()
    runTest1()

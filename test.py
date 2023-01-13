from modelBuilder import *
from model import *
from dataPreparation import *
from dask_ml.model_selection import train_test_split

def genTest():
    X0, X1, Y, train_gen, val_gen, test_gen = prepareDataMixed(lookback=10, lookforward=1, batch_size=1)

    X, Y = train_gen[0]
    print(X[1].shape)


def runTest1():
    X0, X1, Y, train_gen, val_gen, test_gen = prepareDataMixed(lookback=10, lookforward=1, batch_size=1)

    name = 'mixed_test'
    dir = './results/' + name
    model = mixedInputTest((X0, X1), Y)
    model, history = train_model(model, train_gen, val_gen, dir, name)

def runTest2():
    X, Y, train_gen, val_gen, test_gen = prepareData(lookback=10, lookforward=1, batch_size=48)

    name = 'single_test'
    dir = './results/' + name
    model = singleInput(X, Y)
    model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)

def runTest3():
    X, Y, train_gen, val_gen, test_gen = prepareData(lookback=10, lookforward=1, batch_size=48, file='./raw_data/omni_5min_2010.csv')
    name = 'transformer_test'
    dir = './results/' + name
    input_shape = (X.shape[1], X.shape[2])
    output_shape = Y.shape[1]
    model = transformer_model(
        input_shape,
        output_shape,
        head_size=20,
        num_heads=60,
        ff_dim=80,
        num_transformer_blocks=8,
        dense_units=[30, 60, 60, 60, 30],
        dropout=0.3,
        dense_dropout=0.2
    )
    model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)
    stats, *_ = test_model(model, test_gen, dir, name, 'transformer')

def runTest4():
    X, Y, train_gen, val_gen, test_gen = prepareData(lookback=10, lookforward=1, batch_size=48, file='./raw_data/omni_5min_2010.csv')
    name = 'transformer_test_f'
    dir = './results/' + name
    input_shape = (X.shape[1], X.shape[2])
    output_shape = Y.shape[1]
    model = transformer_model_f(
        input_shape,
        output_shape,
        head_size=20,
        num_heads=24,
        ff_dim=50,
        num_transformer_blocks=10,
        dense_units=[480, 240, 120, 60, 30],
        dropout=0.2,
        dense_dropout=0.2
    )
    model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)
    stats, *_ = test_model(model, test_gen, dir, name, 'transformer')

if __name__ == '__main__':
    #genTest()
    #runTest1()
    #runTest2()
    #runTest3()
    runTest4()

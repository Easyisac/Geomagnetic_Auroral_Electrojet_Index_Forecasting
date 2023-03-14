import math
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dask_ml.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.utils import shuffle

import zarr
import gcsfs
import dask
import dask.array as da
import tensorflow as tf

dask.config.set({"array.slicing.split_large_chunks": False})


def create_value_dataset(file='./raw_data/omni_5min_2010_2020.csv', lookback=1, lookforward=1):
    const = 12
    dataset = pd.read_csv(file)
    dataset['date'] = set_date(dataset)
    dataset = clean_missing(dataset, 4, -1)
    columns = dataset.columns.tolist()
    colsX = columns[4:-1]  # + columns[-3:-1]
    colsY = columns[-4]
    dataX = dataset[colsX].to_numpy()
    dataY = dataset[colsY].to_numpy(dtype='float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataX, dataY)
    x = scaled.reshape(scaled.shape[0] // const, const, scaled.shape[1])
    y = dataY.reshape(dataY.shape[0] // const, const)
    X = sliding_window_view(x[:-lookforward], (lookback, x.shape[1], x.shape[2]))
    X = X.reshape(X.shape[0], X.shape[3] * X.shape[4], X.shape[5])
    Y = sliding_window_view(y[lookback:], lookforward, axis=0)
    Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])

    context_embedding = create_context_embedding(X)
    target_embedding = create_target_embedding(Y.shape)
    return X, Y, context_embedding, target_embedding


def create_value_dataset_hourly(file='./raw_data/omny_hourly.csv', lookback=1, lookforward=1):
    const = 1
    dataset = pd.read_csv(file)
    dataset['datetime'] = set_day(dataset)
    dataset = dataset[('2007-12-31' < dataset['datetime'])]
    dataset = dataset[(dataset['datetime'] < '2019-01-01')]
    dataset['datetime'] = dataset['datetime'].apply(lambda d: d.value)
    dataset = clean_missing(dataset, 3, -1)
    columns = dataset.columns.tolist()
    colsX = columns[3:]
    colsY = columns[-4]
    dataX = dataset[colsX].to_numpy()
    dataY = dataset[colsY].to_numpy(dtype='float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataX, dataY)
    x = scaled.reshape(scaled.shape[0] // const, const, scaled.shape[1])
    y = dataY.reshape(dataY.shape[0] // const, const)
    X = sliding_window_view(x[:-lookforward], (lookback, x.shape[1], x.shape[2]))
    X = X.reshape(X.shape[0], X.shape[3] * X.shape[4], X.shape[5])
    Y = sliding_window_view(y[lookback:], lookforward, axis=0)
    Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])

    context_embedding = create_context_embedding(X)
    target_embedding = create_target_embedding(Y.shape)
    return X, Y, context_embedding, target_embedding

def create_value_dataset_hourly_cols(file='./raw_data/omny_hourly.csv', lookback=1, lookforward=1):
    const = 1
    dataset = pd.read_csv(file)
    dataset['datetime'] = set_day(dataset)
    dataset = clean_missing(dataset, 3, -1)
    columns = dataset.columns.tolist()
    colsX = columns[3:-1]
    colsY = columns[-3:-1]
    dataX = dataset[colsX].to_numpy()
    dataY = dataset[colsY].to_numpy(dtype='float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataX, dataY)
    x = scaled.reshape(scaled.shape[0] // const, const, scaled.shape[1])
    y = dataY.reshape(dataY.shape[0] // const, const, dataY.shape[1])
    X = sliding_window_view(x[:-lookforward], (lookback, x.shape[1], x.shape[2]))
    X = X.reshape(X.shape[0], X.shape[3] * X.shape[4], X.shape[5])
    Y = sliding_window_view(y[lookback:], (lookforward, y.shape[1], y.shape[2]))
    Y = Y.reshape(Y.shape[0], Y.shape[3] * Y.shape[4], Y.shape[5])

    # context_embedding = create_context_embedding(X)
    # target_embedding = create_target_embedding(Y.shape)
    return X, Y#, context_embedding, target_embedding

def set_date(data):
    date = data[['Year', 'Day', 'Hour', 'Minute']]
    return pd.to_datetime(date['Year'] * 10000000 + date['Day'] * 10000 + date['Hour'] * 100 + date['Minute'],
                          format='%Y%j%H%M')

def set_day(data):
    date = data[['YEAR', 'DOY', 'Hour']]
    return pd.to_datetime(date['YEAR'] * 100000 + date['DOY'] * 100 + date['Hour'], format='%Y%j%H')


def clean_missing(data, start, end):
    cols = data.columns.tolist()[start:end]
    for col in cols:
        column = data[col]
        if check_errors(column.max()):
            masked = column.mask(column == column.max())
            data[col] = masked.interpolate(limit_direction='both')
    return data


def check_errors(x):
    length = int(math.log10(x)) + 1
    y = (int(math.pow(10, length)) - 1)
    return y <= x


def create_context_embedding(X):
    shape = X.shape[1] * X.shape[2]
    context_embedding = np.zeros((X.shape[0], shape, 3))
    context_embedding[:, :, 1] = list(range(0, X.shape[2])) * X.shape[1]
    context_embedding[:, :, 2] = [i - X.shape[1] for i in range(X.shape[1]) for _ in range(X.shape[2])]
    context_embedding[:, :, 0] = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    return context_embedding


def create_target_embedding(target_shape):
    shape = target_shape[1]
    target_embedding = np.zeros((shape, 3))
    target_embedding[:, 1] = 9
    target_embedding[:, 2] = range(1, shape + 1)
    return target_embedding


def create_image_dataset(lookback=1, lookforward=1):
    const = 10
    t_obs = np.empty(shape=864037, dtype='U32')
    images = da.zeros(shape=(864037, 512, 512), chunks=(1000, -1, -1))
    index = 0
    total = 0
    for i in range(2010, 2021):
        seconds = time.time()
        print('Starting year: {}'.format(i))
        gcs = gcsfs.GCSFileSystem(access="read_only")
        loc = "fdl-sdoml-v2/sdomlv2.zarr/{}".format(i)
        store = gcsfs.GCSMap(loc, gcs=gcs, check=False)
        root = zarr.group(store)
        data = root['304A']  # 304, 1600 would be the best in this order
        tob = np.array(data.attrs["T_OBS"])
        arr = da.from_array(data, chunks=(1000, 512, 512))
        t_obs[index:tob.shape[0] + index] = tob
        images[index:arr.shape[0] + index, :, :] = arr
        index += arr.shape[0]

        seconds2 = time.time()
        elapsed = seconds2 - seconds
        total += elapsed
        print('Ending year: {}, elapsed: {}, total: {} minutes\n'.format(i, elapsed, total / 60))

    t_obs_indexes = t_obs.argsort()
    t_obs = t_obs[t_obs_indexes[::-1]]
    images = images[t_obs_indexes[::-1]]
    print(images)
    print(images.chunks)

    start = time.time()
    df_time = pd.DataFrame(t_obs, index=np.arange(np.shape(t_obs)[0]), columns=["Time"])
    df_time["Time"] = pd.to_datetime(df_time["Time"], utc=True)
    selected_times = pd.date_range(start="2010-05-13 00:00:00", end="2020-12-31 23:59:59", freq="6T", tz="UTC")
    time_index = np.zeros(shape=len(selected_times))
    index = 0
    for i in range(len(selected_times)):
        a = abs(df_time["Time"][index] - selected_times[i])
        b = abs(df_time["Time"][index + 1] - selected_times[i])
        if a < b:
            time_index[i] = index
        else:
            time_index[i] = index + 1
            if index < len(t_obs) - 1:
                index += 1

    print('Time: {}'.format(time.time() - start))
    print('Total time: {} minutes'.format((total + time.time() - start) / 60))
    images = images[time_index, :, :].rechunk(1000, -1, -1)
    print(images)
    print(images.chunks)
    images = images.reshape(images.shape[0] // const, const, images.shape[1], images.shape[2])
    X = da.lib.stride_tricks.sliding_window_view(images[:-lookforward],
                                                 (lookback, images.shape[1], images.shape[2], images.shape[3]))
    X = X.reshape(X.shape[0], X.shape[4] * X.shape[5], X.shape[6], X.shape[7])
    return X


class DataGen(tf.keras.utils.Sequence):

    def __init__(self, X, Y, batch_size, shuffle=False):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = X.shape[0]

    def on_epoch_end(self):
        pass
        # if self.shuffle:
        #     self.X0, self.X1, self.Y = shuffle(self.X0, self.X1, self.Y, random_state=42)

    def __getitem__(self, index):
        bX = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        bY = self.Y[index * self.batch_size:(index + 1) * self.batch_size]
        return bX, bY

    def __len__(self):
        return self.n // self.batch_size


class MixedDataGen(tf.keras.utils.Sequence):

    def __init__(self, X0, X1, Y, batch_size, shuffle=False):
        self.X0 = X0
        self.X1 = X1.rechunk(chunks=(1000, -1, -1, -1))
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = X0.shape[0]
        self.X1_newShape = (self.batch_size, self.X1.shape[1], self.X1.shape[2], self.X1.shape[3])
        print(self.X1)
        print(self.X1.chunks)

    def on_epoch_end(self):
        pass
        # if self.shuffle:
        #     self.X0, self.X1, self.Y = shuffle(self.X0, self.X1, self.Y, random_state=42)

    def __getitem__(self, index):
        bX0 = self.X0[index * self.batch_size:(index + 1) * self.batch_size]
        bX1 = np.zeros(self.X1_newShape)
        da.store(self.X1[index * self.batch_size:(index + 1) * self.batch_size], bX1)
        bY = self.Y[index * self.batch_size:(index + 1) * self.batch_size]
        return (bX0, bX1), bY

    def __len__(self):
        return self.n // self.batch_size


class DataGenFull(tf.keras.utils.Sequence):

    def __init__(self, X, target, Y, batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.target = np.broadcast_to(target, (self.n,) + target.shape)

    def __getitem__(self, index):
        bX = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        bTarget = self.target[index * self.batch_size:(index + 1) * self.batch_size]
        bY = self.Y[index * self.batch_size:(index + 1) * self.batch_size]
        return (bX, bTarget), bY

    def __len__(self):
        return self.n // self.batch_size


class DataGenTuner(tf.keras.utils.Sequence):

    def __init__(self, X, target, Y, batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.target = np.broadcast_to(target, (self.n,) + target.shape)

    def on_epoch_end(self):
        self.indexes = np.arange(self.n)
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        bX = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        bTarget = self.target[index * self.batch_size:(index + 1) * self.batch_size]
        bY = self.Y[index * self.batch_size:(index + 1) * self.batch_size]
        return (bX, bTarget), bY

    def __len__(self):
        return 500


class EmbeddedDataGen(tf.keras.utils.Sequence):

    def __init__(self, context_embedding, target_embedding, Y, batch_size):
        self.context = context_embedding
        self.Y = Y
        self.batch_size = batch_size
        self.n = self.context.shape[0]
        self.target = np.broadcast_to(target_embedding, (self.n,) + target_embedding.shape)

    def __getitem__(self, index):
        bX = self.context[index * self.batch_size:(index + 1) * self.batch_size]
        bTarget = self.target[index * self.batch_size:(index + 1) * self.batch_size]
        bY = self.Y[index * self.batch_size:(index + 1) * self.batch_size]
        return (bX, bTarget), bY

    def __len__(self):
        return self.n // self.batch_size


def prepare_data(lookback=1, lookforward=1, batch_size=1, file='./raw_data/omni_5min_2010_2020.csv'):
    X, Y, _, _ = create_value_dataset(lookback=lookback, lookforward=lookforward, file=file)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=42,
                                                    shuffle=False)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.01,
                                                  random_state=42, shuffle=False)
    train_gen = DataGen(Xtrain, Ytrain, batch_size)
    val_gen = DataGen(Xval, Yval, batch_size)
    test_gen = DataGen(Xtest, Ytest, batch_size)
    return X, Y, train_gen, val_gen, test_gen


def prepare_data_hours(lookback=1, lookforward=1, batch_size=1, file='./raw_data/omni_5min_2010_2020.csv'):
    X, Y, _, _ = create_value_dataset_hourly(lookback=lookback, lookforward=lookforward, file=file)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=42,
                                                    shuffle=True)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.01,
                                                  random_state=42, shuffle=True)
    train_gen = DataGen(Xtrain, Ytrain, batch_size)
    val_gen = DataGen(Xval, Yval, batch_size)
    test_gen = DataGen(Xtest, Ytest, batch_size)
    return X, Y, train_gen, val_gen, test_gen


def prepare_data_mixed(lookback=1, lookforward=1, batch_size=1):
    X0, Y, _, _ = create_value_dataset(lookback=lookback, lookforward=lookforward)
    X1 = create_image_dataset(lookback=lookback, lookforward=lookforward)
    X0train, X0test, X1train, X1test, Ytrain, Ytest = train_test_split(X0, X1, Y, test_size=0.1, random_state=42,
                                                                       shuffle=False)
    X0train, X0val, X1train, X1val, Ytrain, Yval = train_test_split(X0train, X1train, Ytrain, test_size=0.01,
                                                                    random_state=42, shuffle=False)
    train_gen = MixedDataGen(X0train, X1train, Ytrain, batch_size)
    val_gen = MixedDataGen(X0val, X1val, Yval, batch_size)
    test_gen = MixedDataGen(X0test, X1test, Ytest, batch_size)
    return X0, X1, Y, train_gen, val_gen, test_gen


def prepare_data_full(lookback=1, lookforward=1, batch_size=1, file='./raw_data/omni_5min_2010_2020.csv'):
    X, Y, _, _ = create_value_dataset(lookback=lookback, lookforward=lookforward, file=file)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=42,
                                                    shuffle=False)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.01,
                                                  random_state=42, shuffle=False)
    target = np.zeros((Y.shape[1], 1))
    train_gen = DataGenFull(Xtrain, target, Ytrain, batch_size)
    val_gen = DataGenFull(Xval, target, Yval, batch_size)
    test_gen = DataGenFull(Xtest, target, Ytest, batch_size)
    return X, Y, train_gen, val_gen, test_gen


def prepare_data_full_hours(lookback=1, lookforward=1, batch_size=1, file='./raw_data/omni_hourly.csv'):
    X, Y, *_ = create_value_dataset_hourly(lookback=lookback, lookforward=lookforward, file=file)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=42,
                                                    shuffle=True)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.01,
                                                  random_state=42, shuffle=True)
    target = np.ones((1, Y.shape[1]))
    train_gen = DataGenFull(Xtrain, target, Ytrain, batch_size)
    val_gen = DataGenFull(Xval, target, Yval, batch_size)
    test_gen = DataGenFull(Xtest, target, Ytest, batch_size)
    return X, Y, train_gen, val_gen, test_gen


def prepare_data_full_hours_cols(lookback=1, lookforward=1, batch_size=1, file='./raw_data/omni_hourly.csv'):
    X, Y = create_value_dataset_hourly_cols(lookback=lookback, lookforward=lookforward, file=file)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=42,
                                                    shuffle=False)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.01,
                                                  random_state=42, shuffle=False)
    target = np.ones((Y.shape[1], Y.shape[2]))
    train_gen = DataGenFull(Xtrain, target, Ytrain, batch_size)
    val_gen = DataGenFull(Xval, target, Yval, batch_size)
    test_gen = DataGenFull(Xtest, target, Ytest, batch_size)
    return X, Y, train_gen, val_gen, test_gen


def prepare_data_embedded(lookback=1, lookforward=1, batch_size=1, file='./raw_data/omni_5min_2010_2020.csv'):
    X, Y, context, target = create_value_dataset(lookback=lookback, lookforward=lookforward, file=file)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(context, Y, test_size=0.1, random_state=42,
                                                    shuffle=False)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.01,
                                                  random_state=42, shuffle=False)
    train_gen = EmbeddedDataGen(Xtrain, target, Ytrain, batch_size)
    val_gen = EmbeddedDataGen(Xval, target, Yval, batch_size)
    test_gen = EmbeddedDataGen(Xtest, target, Ytest, 1)
    return X, Y, train_gen, val_gen, test_gen

def prepare_data_hours_tuner(lookback=1, lookforward=1, batch_size=1, file='./raw_data/omni_hourly.csv'):
    X, Y, *_ = create_value_dataset_hourly(lookback=lookback, lookforward=lookforward, file=file)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=42,
                                                    shuffle=True)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.01,
                                                  random_state=42, shuffle=True)
    target = np.ones((1, Y.shape[1]))
    train_gen = DataGenTuner(Xtrain, target, Ytrain, batch_size)
    val_gen = DataGenFull(Xval, target, Yval, batch_size)
    test_gen = DataGenFull(Xtest, target, Ytest, batch_size)
    return X, Y, train_gen, val_gen, test_gen
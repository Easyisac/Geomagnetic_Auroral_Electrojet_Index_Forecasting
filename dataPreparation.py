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



def create_value_dataset(file='raw_data/omni_5min_2010_2020.csv', loockback=1, loockforward=1):
    const = 12
    dataset = pd.read_csv(file)
    dataset['datetime'] = set_date(dataset)
    columns = dataset.columns.tolist()
    colsX = columns[4:-4] + columns[-3:-1]
    colsY = columns[-4]
    dataX = dataset[colsX].to_numpy()
    dataY = dataset[colsY].to_numpy(dtype='float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataX, dataY)
    x = scaled.reshape(scaled.shape[0] // const, const, scaled.shape[1])
    y = dataY.reshape(dataY.shape[0] // const, const)
    X = sliding_window_view(x[:-loockforward], (loockback, x.shape[1], x.shape[2]))
    X = X.reshape(X.shape[0], X.shape[3] * X.shape[4], X.shape[5])
    Y = sliding_window_view(y[loockback:], loockforward, axis=0)
    Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])
    print(X.shape, Y.shape)
    return X, Y


def set_date(data):
    date = data[['Year', 'Day', 'Hour', 'Minute']]
    return pd.to_datetime(date['Year'] * 10000000 + date['Day'] * 10000 + date['Hour'] * 100 + date['Minute'],
                          format='%Y%j%H%M')


def clean_missing(data):
    cols = data.columns.tolist()[1:]
    for col in cols:
        column = data[col]
        if check_errors(column.max()):
            masked = column.mask(column == column.max())
            data[col] = masked.interpolate()
    return data


def check_errors(x):
    length = int(math.log10(x)) + 1
    y = (int(math.pow(10, length)) - 1)
    return y < x


def create_image_dataset(loockback=1, loockforward=1):
    const = 10
    t_obs = np.empty(shape=864037, dtype='U32')
    images = da.zeros(shape=(864037, 512, 512))
    index = 0
    for i in range(2010, 2021):
        seconds = time.time()
        local_time = time.ctime(seconds)
        print('Starting year: {} , time: {}'.format(i, local_time))
        gcs = gcsfs.GCSFileSystem(access="read_only")
        loc = "fdl-sdoml-v2/sdomlv2.zarr/{}".format(i)
        store = gcsfs.GCSMap(loc, gcs=gcs, check=False)
        root = zarr.group(store)
        data = root['171A']
        tob = np.array(data.attrs["T_OBS"])
        arr = da.from_array(data)
        t_obs[index:tob.shape[0] + index] = tob
        images[index:arr.shape[0] + index, :, :] = arr
        index += arr.shape[0]

        seconds2 = time.time()
        local_time = time.ctime(seconds2)
        print('Ending year: {} , time: {}, elapsed: {}\n'.format(i, local_time, seconds2 - seconds))

    t_obs_indexes = t_obs.argsort()
    t_obs = t_obs[t_obs_indexes[::-1]]
    images = images[t_obs_indexes[::-1]]

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

    images = images[time_index, :, :]
    images = images.reshape(images.shape[0] // const, const, images.shape[1], images.shape[2])
    X = sliding_window_view(images[:-loockforward], (loockback, images.shape[1], images.shape[2], images.shape[3]))
    X = X.reshape(X.shape[0], X.shape[4] * X.shape[5], X.shape[6], X.shape[7])
    print(X.shape)
    return X


class DataGen(tf.keras.utils.Sequence):

    def __init__(self, X0, X1, Y, batch_size, shuffle=True):
        self.X0 = X0
        self.X1 = X1
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = X0.shape[0]

    def on_epoch_end(self):
        if self.shuffle:
            self.X0, self.X1, self.Y = shuffle(self.X0, self.X1, self.Y, random_state=42)

    def __getitem__(self, index):
        bX0 = self.X0[index * self.batch_size:(index + 1) * self.batch_size]
        bX1 = self.X1[index * self.batch_size:(index + 1) * self.batch_size]
        bY = self.Y[index * self.batch_size:(index + 1) * self.batch_size]
        return (bX0, bX1), bY

    def __len__(self):
        return self.n // self.batch_size


import math
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view

import zarr
import gcsfs
import dask.array as da

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sunpy.visualization.colormaps as cm


def create_value_dataset(file='raw_data/omni_5min_2010.csv', loockback=1, loockforward=1):
    const = 12

    dataset = pd.read_csv(file)
    dataset['datetime'] = set_date(dataset)
    columns = dataset.columns.tolist()
    colsX = columns[4:-4] + columns[-3:-1]
    colsY = columns[-4]
    dataX = dataset[colsX].to_numpy()
    dataY = dataset[colsY].to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataX, dataY)
    # lines = scaled.shape[0] - loockback * const - loockforward * const - 1
    # size = scaled.shape[1]
    # X = np.zeros((lines, loockback * const, size))
    # Y = np.zeros((lines, loockforward * const, 1))
    # for i in range(lines):
    #     X[i] = scaled[i:i + loockback * const]
    #     Y[i] = dataY[i + loockback * const: i + loockback * const + loockforward * const]
    print(dataY.shape)
    X = sliding_window_view(scaled[:-loockforward*const], loockback*const, axis=0)
    Y = sliding_window_view(dataY[loockback*const:], loockforward*const)
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


def create_image_dataset():
    t_obs = np.empty(shape=(0,))
    images = np.empty(shape=(0, 512, 512))
    for i in range(2010, 2021):
        seconds = time.time()
        local_time = time.ctime(seconds)
        print('Starting year: {} , time: {}'.format(i, local_time))
        gcs = gcsfs.GCSFileSystem(access="read_only")
        loc = "fdl-sdoml-v2/sdomlv2.zarr/{}".format(i)
        store = gcsfs.GCSMap(loc, gcs=gcs, check=False)
        root = zarr.group(store)
        data = root['171A']
        print(t_obs.shape)
        print(images.shape)
        t_obs = np.append(t_obs, np.array(data.attrs["T_OBS"]), axis=0)
        images = np.append(images, da.from_array(data), axis=0)
        print(t_obs.shape)
        print(images.shape)

        seconds2 = time.time()
        local_time = time.ctime(seconds2)
        print('Ending year: {} , time: {}, elapsed: {}'.format(i, local_time, seconds2 - seconds))

    # gcs = gcsfs.GCSFileSystem(access="read_only")
    # loc = "fdl-sdoml-v2/sdomlv2.zarr/2020"
    # store = gcsfs.GCSMap(loc, gcs=gcs, check=False)
    # root = zarr.group(store)
    # data = (root['171A'])
    # t_obs = np.append(t_obs, np.array(data.attrs["T_OBS"]))
    # images = np.append(images, da.from_array(data))

    print(len(t_obs))
    print(sorted(t_obs)[0])
    print(sorted(t_obs)[-1])

    start = time.time()
    df_time = pd.DataFrame(t_obs, index=np.arange(np.shape(t_obs)[0]), columns=["Time"])
    df_time["Time"] = pd.to_datetime(df_time["Time"], utc=True)
    selected_times = pd.date_range(start="2010-05-13 00:00:00", end="2020-12-31 23:59:59", freq="6T", tz="UTC")
    # print(len(selected_times))
    selected_index = []
    for n, i in enumerate(selected_times):
        if n % 10000 == 0:
            print(n)
        selected_index.append(np.argmin(abs(i - df_time["Time"])))
    time_index = list(filter(lambda x: x >= 0, selected_index))
    print(len(time_index))
    images = da.from_array(data)[time_index, :, :]
    image = images[0, :, :]
    plt.figure(figsize=(10, 10))
    colormap = plt.get_cmap('sdoaia171')
    plt.imshow(image, origin='lower', vmin=10, vmax=1000, cmap=colormap)
    plt.show()
    end = time.time()
    print('Time for dataset: {} minutes'.format((end - start) / 60))


def create_image_dataset2():
    t_obs = np.empty(shape=(0,))
    images = np.empty(shape=(0, 512, 512))

    for i in range(2011, 2021):
        seconds = time.time()
        local_time = time.ctime(seconds)
        print('Starting year: {} , time: {}'.format(i, local_time))
        gcs = gcsfs.GCSFileSystem(access="read_only")
        loc = "fdl-sdoml-v2/sdomlv2.zarr/{}".format(i)
        store = gcsfs.GCSMap(loc, gcs=gcs, check=False)
        root = zarr.group(store)
        data = root['171A']
        print(t_obs.shape)
        print(images.shape)
        t_obs = np.append(t_obs, np.array(data.attrs["T_OBS"]), axis=0)
        images = np.append(images, da.from_array(data), axis=0)
        print(t_obs.shape)
        print(images.shape)

        seconds2 = time.time()
        local_time = time.ctime(seconds2)
        print('Ending year: {} , time: {}, elapsed: {}'.format(i, local_time, seconds2 - seconds))

        start = time.time()
        df_time = pd.DataFrame(t_obs, index=np.arange(np.shape(t_obs)[0]), columns=["Time"])
        df_time["Time"] = pd.to_datetime(df_time["Time"], utc=True)
        selected_times = pd.date_range(start="{}-01-01 00:00:00".format(i), end="{}-12-31 23:59:59".format(i),
                                       freq="6T", tz="UTC")
        selected_index = []
        for n, i in enumerate(selected_times):
            if n % 10000 == 0:
                print(n)
            selected_index.append(np.argmin(abs(i - df_time["Time"])))
        time_index = list(filter(lambda x: x >= 0, selected_index))
        print(len(time_index))
        np.append(images, da.from_array(data)[time_index, :, :])
        end = time.time()
        print('Time for dataset: {} minutes'.format((end - start) / 60))
    image = images[0, :, :]
    plt.figure(figsize=(10, 10))
    colormap = plt.get_cmap('sdoaia171')
    plt.imshow(image, origin='lower', vmin=10, vmax=1000, cmap=colormap)
    plt.show()


def create_image_dataset3(loockback=1, loockforward=1):
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

    start = time.time()
    t_obs_indexes = t_obs.argsort()
    t_obs = t_obs[t_obs_indexes[::-1]]
    images = images[t_obs_indexes[::-1]]
    print('Sorting time: {}'.format(time.time() - start))

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
    print(time_index.shape)
    images = images[time_index, :, :]
    # lines = time_index.shape[0] - loockback * const - loockforward * const - 1
    # X_index = np.zeros(shape=(lines, loockback * const))
    # X = da.zeros(shape=(lines, loockback * const, 512, 512))
    # for i in range(lines):
    #     if i%10000 == 0:
    #         print(i, 'index')
    #     X_index[i] = time_index[i:i + loockback * const]
    # for i in range(lines):
    #     if i%10000 == 0:
    #         print(i)
    #     X[i] = images[X_index[i], :, :]
    # return X
    X = sliding_window_view(images, loockback*const, axis=0)
    print(X.shape)
    return X


# create_value_dataset()
create_image_dataset3()

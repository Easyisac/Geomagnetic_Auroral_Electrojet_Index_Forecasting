import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import zarr
import gcsfs
import dask.array as da

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sunpy.visualization.colormaps as cm

def create_value_dataset(file='raw_data/omni_5min_2010.csv'):
    dataset = pd.read_csv(file)
    dataset['datetime'] = set_date(dataset)
    cols = dataset.columns.tolist()
    cols = cols[-1:] + cols[4:-1]
    dataset = dataset[cols]
    dataset = clean_missing(dataset)


def set_date(data):
    date = data[['Year', 'Day', 'Hour', 'Minute']]
    return pd.to_datetime(date['Year'] * 10000000 + date['Day'] * 10000 + date['Hour'] * 100 + date['Minute'],
                          format='%Y%j%H%M')


def clean_missing(data):
    cols = data.columns.tolist()[1:]
    print(cols)
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
    gcs = gcsfs.GCSFileSystem(access="read_only")
    loc = "fdl-sdoml-v2/sdomlv2.zarr/2012"
    store = gcsfs.GCSMap(loc, gcs=gcs, check=False)
    root = zarr.group(store)
    data = root['171A']
    all_image = da.from_array(data)
    print(all_image)
    t_obs = np.array(data.attrs["T_OBS"])
    print(len(t_obs))
    print(sorted(t_obs)[0])
    print(sorted(t_obs)[-1])
    # df_time = pd.DataFrame(t_obs, index=np.arange(np.shape(t_obs)[0]), columns=["Time"])
    # df_time["Time"] = pd.to_datetime(df_time["Time"])
    # selected_times = pd.date_range(start="2010-01-01 00:00:00", end="2010-12-31 23:59:59", freq="5T", tz="UTC")
    # print(selected_times)
    # selected_index = []
    # for i in selected_times:
    #     selected_index.append(np.argmin(abs(df_time["Time"] - i)))
    # time_index = list(filter(lambda x: x > 0, selected_index))
    # print(len(time_index))
    # images = da.from_array(data)[time_index, :, :]
    # image=images[0,:,:]
    # plt.figure(figsize=(10,10))
    # colormap = plt.get_cmap('sdoaia171')
    # plt.imshow(image,origin='lower',vmin=10,vmax=1000,cmap=colormap)
    # plt.show()


create_image_dataset()

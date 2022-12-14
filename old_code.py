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
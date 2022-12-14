import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def raw_mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def raw_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def raw_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def raw_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def raw_pearson(y_true, y_pred):
    avg_true = np.average(y_true)
    avg_pred = np.average(y_pred)
    diffprod = 0
    truediff = 0
    preddiff = 0
    for i in range(len(y_true)):
        true = y_true[i] - avg_true
        pred = y_pred[i] - avg_pred
        truediff += true ** 2
        preddiff += pred ** 2
        diffprod += true * pred
    return diffprod / np.math.sqrt(truediff * preddiff)


def calculate_stats(actual, pred):
    mse = raw_mse(actual, pred)
    mae = raw_mae(actual, pred)
    rmse = raw_rmse(actual, pred)
    mape = raw_mape(actual, pred)
    pear = raw_pearson(actual, pred)

    return [mse, mae, rmse, mape, pear]


def draw_chart(data, stats, dir, name, label):
    plt.clf()
    fig = plt.figure(figsize=(10, 8), dpi=80)
    sns.set_theme(style="darkgrid")
    [mse, mae, rmse, rmape] = stats
    text = "Overall Test Performance:\nMSE: %f, MAE: %f, RMSE: %f, MAPE: %f" % (
        mse, mae, rmse, rmape)
    fig.suptitle(text, fontsize=15)
    # plt.plot(act, 'r-', label='actual values', linewidth=0.5)
    # plt.plot(pred, 'b-.', label=label, linewidth=0.7)
    sns.lineplot(x='time', y='actual', data=data)
    sns.lineplot(x='time', y='predicted', data=data)
    plt.legend(loc='upper right')
    path = dir + '\\' + name + '.png'
    plt.savefig(path)
    plt.close(fig)

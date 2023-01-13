from datetime import datetime
import tensorflow.keras
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
import tensorflow.keras.backend as bk
import numpy as np
from tools import rmse, calculate_stats, draw_chart

train_epochs = 100

def execute_model(model, train_data, val_data, test_data, dir, model_name):

    bk.clear_session()
    np.random.seed(1)

    model.summary()
    model, history = train_model(model, train_data, val_data, dir, model_name)
    results = evaluate_model(model, test_data)

    return model, history, results


def train_model(model, data, validation, dir, model_name):
    bk.clear_session()
    model.compile(optimizer='adam', loss=rmse, metrics=['mae', 'mse', rmse])
    model.summary()
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    log_dir = "{}/logs/fit/".format(dir) + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, update_freq="epoch")
    history = model.fit(x=data, epochs=train_epochs, validation_data=validation, callbacks=[es, tensorboard], max_queue_size=0)

    #plot_model(model, to_file="{}/plot_{}.png".format(dir, model_name), show_shapes=True, show_layer_names=True)

    plt.clf()
    plt.title('Rooted Mean Squared Error of {}'.format(model_name))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig("{}/loss.png".format(dir))

    plt.clf()
    plt.title('Mean Absolute Error of {}'.format(model_name))
    plt.plot(history.history['mae'], label='train')
    plt.plot(history.history['val_mae'], label='test')
    plt.legend()
    plt.savefig("{}/mae.png".format(dir))

    return model, history


def evaluate_model(model, data):

    results = model.evaluate(data, verbose=1)
    print("[loss, mse, mae, rmse]")
    print(results)
    return results


def test_model(model, data, dir, name, label):
    x = data.X
    y = data.Y
    y_pred = model.predict(x, verbose=1)
    print(len(y_pred), len(y))
    print("___________ calculate stats ___________")
    print("[MSE, MAE, RMSE, MAPE]")
    stats = calculate_stats(y, y_pred)
    print(stats)
    #draw_chart(np.asarray(y), np.asarray(y_pred), stats, dir, name, label)
    for i in range(y.shape[1]):
        par_stats = calculate_stats(y[:,i], y_pred[:,i])
        print("[MSE, MAE, RMSE, MAPE] of {}".format(i))
        print(par_stats)

    return stats, y, y_pred

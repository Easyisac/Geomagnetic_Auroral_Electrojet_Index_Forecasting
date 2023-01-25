from datetime import datetime
import tensorflow.keras
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
import tensorflow.keras.backend as bk
import numpy as np
from tools import rmse, mape, calculate_stats, draw_chart
from tensorflow import keras
from keras.optimizer_v2.adam import Adam

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
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-1, decay_steps=100000, decay_rate=0.9, staircase=True
    )
    optimizer = Adam(learning_rate=lr_scheduler)
    model.compile(optimizer=optimizer, loss=rmse, metrics=['mae', 'mse', rmse, mape])
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    log_dir = "{}/logs/fit/".format(dir) + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, update_freq="epoch")
    history = model.fit(x=data, epochs=train_epochs, validation_data=validation, callbacks=[es, tensorboard],
                        max_queue_size=0)

    # plot_model(model, to_file="{}/plot_{}.png".format(dir, model_name), show_shapes=True, show_layer_names=True)

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
    print("[loss, mse, mae, rmse, mape]")
    print(results)
    return results


def test_model(model, data, dir, name, label):
    x = data.X
    target = data.target
    y = data.Y
    y_pred = model.predict(((x, target)), verbose=1)
    print("___________ calculate stats ___________")
    print("[MSE, MAE, RMSE, MAPE]")
    stats = calculate_stats(y, y_pred)
    print(stats)
    draw_chart(np.asarray(y[:, 0]), np.asarray(y_pred[:, 0]), stats, dir, name, label)
    print(y.shape, y_pred.shape)
    for i in range(y.shape[1]):
        par_stats = calculate_stats(y[:, i], y_pred[:, i])
        print("[MSE, MAE, RMSE, MAPE] of {}".format(i))
        print(par_stats)

    return stats, y, y_pred

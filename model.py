from datetime import datetime
import tensorflow.keras
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
import tensorflow.keras.backend as bk
import numpy as np
from tools import rmse, calculate_stats, draw_chart

train_epochs = 100

# def execute_model(model, train_data, test_data, dir, model_name):
#
#     bk.clear_session()
#     np.random.seed(1)
#
#     model.summary()
#     model, history = train_model(model, X_train, Y_train, dir, model_name)
#     results = evaluate_model(model, X_test, Y_test)
#
#     return model, history, results


def train_model(model, data, validation, dir, model_name):

    model.compile(optimizer='adam', loss=rmse, metrics=['mae', 'mse', rmse])

    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    log_dir = "{}/logs/fit/".format(dir) + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, update_freq="epoch")
    history = model.fit(x=data, epochs=train_epochs, validation_data=validation, callbacks=[es, tensorboard])

    plot_model(model, to_file="{}/plot_{}.png".format(dir, model_name), show_shapes=True, show_layer_names=True)

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


# def evaluate_model(model, X_test, Y_test):
#
#     results = model.evaluate(X_test, Y_test, verbose=1, batch_size=X_test.shape[0])
#     print("[loss, mse, mae, rmse]")
#     print(results)
#     return results


# def test_model(model, X_test, Y_test, scaler, dir, name, label):
#     """
#     Performs an insight evaluation on a model
#     :param model: already compiled and trained model to test
#     :param X_test: x for testing
#     :param Y_test: y for testing
#     :param scaler: scaler to rescale results
#     :param dir: directory to save results
#     :param name: name of files to save
#     :param label: label for graphs
#     :return: stats of evaluation and actual y and predicted ones
#     """
#     y_pred = model.predict(X_test, verbose=1)
#     y_pred_scaled = scaler.inverse_transform(y_pred)
#     y_scaled = scaler.inverse_transform(np.squeeze(Y_test, axis=2))
#     print("___________ calculate stats ___________")
#     print("[MSE, MAE, RMSE, MAPE]")
#     stats = calculate_stats(np.asarray(y_scaled), np.asarray(y_pred_scaled))
#     print(stats)
#     draw_chart(y_scaled, y_pred_scaled, stats, dir, name, label)
#
#     return stats, np.asarray(y_scaled), np.asarray(y_pred_scaled)

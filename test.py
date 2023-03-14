import numpy as np

import dataPreparation
from modelBuilder import *
from fullTransformerBuilder import full_transformer_model
from model import *
from dataPreparation import *
from dask_ml.model_selection import train_test_split
import keras_tuner
import winsound



def runTest():
    X, Y, train_gen, val_gen, test_gen = prepare_data_full_hours(lookback=120, lookforward=24, batch_size=16,
                                                                 file='./raw_data/omni_new.csv')
    name = 'full_transformer_test'
    dir = './results/' + name
    encoder_shape = (X.shape[1], X.shape[2])
    decoder_shape = (1, Y.shape[1])
    model = full_transformer_model(
        encoder_shape,
        decoder_shape,
        e_head_size=120,                     #120,
        e_num_heads=4,                      #4,
        e_ff_dim=180,                       #120,
        num_encoder_blocks=6,               #6,
        d_head_size=96,                     #20,
        d_num_heads=4,                      #4,
        d_head_size_cross=120,               #24,
        d_num_heads_cross=8,                #4,
        d_ff_dim=48,                       #24,
        num_decoder_blocks=6,               #6,
        dense_units=[24, 48, 96, 96, 48, 24],
        e_dropout=0.36,                    #0.1,
        d_dropout=0.08,                      #0.1,
        dense_dropout=0.015                  #0.1
    )
    model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)
    stats, *_ = test_model(model, test_gen, dir, name, 'full_transformer')




def runTest2():
    X, Y, train_gen, val_gen, test_gen = prepare_data_hours(lookback=48, lookforward=24, batch_size=16,
                                                                 file='./raw_data/omni_new.csv')
    name = 'classic_lstm'
    dir = './results/' + name
    input_shape = (X.shape[1], X.shape[2])
    output_shape = Y.shape[1]
    model = classic_LSTM(
        input_shape,
        output_shape,
        lstm_nodes=48
    )
    model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)
    stats, *_ = test_model(model, test_gen, dir, name, 'classic_lstm')


def build_model(hp):
    encoder_shape = (120, 21)
    decoder_shape = (1, 24)
    model = full_transformer_model(
        encoder_shape,
        decoder_shape,
        e_head_size=hp.Int("e_head_size", min_value=10, max_value=240, step=10),
        e_num_heads=hp.Int("e_num_heads", min_value=1, max_value=8, step=1),
        e_ff_dim=hp.Int("e_ff_dim", min_value=20, max_value=240, step=10),
        num_encoder_blocks=hp.Int("num_encoder_blocks", min_value=1, max_value=8, step=1),
        d_head_size=hp.Int("d_head_size", min_value=10, max_value=120, step=10),
        d_num_heads=hp.Int("d_num_heads", min_value=1, max_value=8, step=1),
        d_head_size_cross=hp.Int("d_head_siz_cross", min_value=10, max_value=120, step=10),
        d_num_heads_cross=hp.Int("d_num_heads_cross", min_value=1, max_value=8, step=1),
        d_ff_dim=hp.Int("d_ff_dim", min_value=20, max_value=240, step=10),
        num_decoder_blocks=hp.Int("num_decoder_blocks", min_value=1, max_value=8, step=1),
        dense_units=[24, 48, 96, 48, 24],
        e_dropout=hp.Float("e_dropout", min_value=0, max_value=0.5),
        d_dropout=hp.Float("d_dropout", min_value=0, max_value=0.5),
        dense_dropout=hp.Float("dense_dropout", min_value=0, max_value=0.5),
    )
    bk.clear_session()
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log"), decay_steps=100000, decay_rate=hp.Float("dr", min_value=0.75, max_value=1, sampling="log"), staircase=True
    )
    optimizer = Adam(learning_rate=lr_scheduler)
    model.compile(optimizer=optimizer, loss=rmse, metrics=['mae', 'mse', rmse])
    return model


def runTest3():
    X, Y, train_gen, val_gen, test_gen = prepare_data_hours_tuner(lookback=120, lookforward=24, batch_size=16,
                                                                 file='./raw_data/omni_new.csv')

    name = 'full_transformer_test'
    dir = './results/' + name

    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_loss",
        max_trials=100,
        executions_per_trial=2,
        overwrite=True,
        directory=dir,
        project_name=name,
    )

    tuner.search_space_summary()
    tuner.search(train_gen, epochs=10, validation_data=val_gen)

    # model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)
    # stats, *_ = test_model(model, test_gen, dir, name, 'full_transformer')



if __name__ == '__main__':
    # runTest()
    runTest2()
    # runTest3()

    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)

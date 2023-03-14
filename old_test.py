def runTest8():
    X, Y, train_gen, val_gen, test_gen = prepare_data_full_hours_cols(lookback=120, lookforward=24, batch_size=16,
                                                                      file='./raw_data/omni_new_2010.csv')
    name = 'full_transformer_test'
    dir = './results/' + name
    encoder_shape = (X.shape[1], X.shape[2])
    decoder_shape = (Y.shape[1], Y.shape[2])
    model = full_transformer_model(
        encoder_shape,
        decoder_shape,
        e_head_size=120,
        e_num_heads=8,
        e_ff_dim=120,
        num_encoder_blocks=8,
        d_head_size=20,
        d_num_heads=4,
        d_ff_dim=24,
        num_decoder_blocks=8,
        dense_units=[24, 48, 96, 192, 96, 48, 24],
        e_dropout=0.1,
        d_dropout=0.1,
        dense_dropout=0.1
    )
    model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)
    stats, y, y_pred = test_model(model, test_gen, dir, name, 'full_transformer')
    y_tot = y[:, :, 1] - y[:, :, 0]
    y_pred_tot = y_pred[:, :, 1] - y_pred[:, :, 0]
    print('---------------------------------------------------------------------------')
    print(y_tot.shape, y_pred_tot.shape)
    stats = calculate_stats(y_tot, y_pred_tot)
    print("[MSE, MAE, RMSE]")
    print(stats)
    stats = calculate_stats(y[:, :, 0], y_pred[:, :, 0])
    print("[MSE, MAE, RMSE]")
    print(stats)
    stats = calculate_stats(y[:, :, 1], y_pred[:, :, 1])
    print("[MSE, MAE, RMSE]")
    print(stats)
    # for i in range(y_tot.shape[1]):
    #     par_stats = calculate_stats(y_tot[:, i], y_pred_tot[:, i])
    #     print("[MSE, MAE, RMSE] of {}".format(i))
    #     print(par_stats)

    def genTest():
        X0, X1, Y, train_gen, val_gen, test_gen = prepare_data_mixed(lookback=10, lookforward=1, batch_size=1)

    X, Y = train_gen[0]
    print(X[1].shape)


def runTest1():
    X0, X1, Y, train_gen, val_gen, test_gen = prepare_data_mixed(lookback=10, lookforward=1, batch_size=1)

    name = 'mixed_test'
    dir = './results/' + name
    model = mixedInputTest((X0, X1), Y)
    model, history = train_model(model, train_gen, val_gen, dir, name)


def runTest2():
    X, Y, train_gen, val_gen, test_gen = prepare_data(lookback=10, lookforward=1, batch_size=48)

    name = 'single_test'
    dir = './results/' + name
    model = singleInput(X, Y)
    model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)


def runTest3():
    X, Y, train_gen, val_gen, test_gen = prepare_data(lookback=10, lookforward=1, batch_size=48,
                                                      file='./raw_data/omni_5min_2010.csv')
    name = 'transformer_test'
    dir = './results/' + name
    input_shape = (X.shape[1], X.shape[2])
    output_shape = Y.shape[1]
    model = transformer_model(
        input_shape,
        output_shape,
        head_size=20,
        num_heads=60,
        ff_dim=80,
        num_transformer_blocks=8,
        dense_units=[30, 60, 60, 60, 30],
        dropout=0.3,
        dense_dropout=0.2
    )
    model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)
    stats, *_ = test_model(model, test_gen, dir, name, 'transformer')


def runTest4():
    X, Y, train_gen, val_gen, test_gen = prepare_data(lookback=10, lookforward=1, batch_size=48,
                                                      file='./raw_data/omni_5min_2010.csv')
    name = 'transformer_test_f'
    dir = './results/' + name
    input_shape = (X.shape[1], X.shape[2])
    output_shape = Y.shape[1]
    model = transformer_model_f(
        input_shape,
        output_shape,
        head_size=60,
        num_heads=12,
        ff_dim=50,
        num_transformer_blocks=10,
        dense_units=[960, 480, 240, 120, 60, 30],
        dropout=0.2,
        dense_dropout=0.2
    )
    model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)
    stats, *_ = test_model(model, test_gen, dir, name, 'transformer')


def runTest5():
    X, Y, train_gen, val_gen, test_gen = prepare_data_embedded(lookback=10, lookforward=1, batch_size=48,
                                                               file='./raw_data/omni_5min_2010.csv')
    name = 'full_transformer_test_f'
    dir = './results/' + name
    encoder_shape = (X.shape[1] * X.shape[2], 3)
    decoder_shape = (Y.shape[1], 3)
    model = full_transformer_model(
        encoder_shape,
        decoder_shape,
        e_head_size=30,
        e_num_heads=2,
        e_ff_dim=50,
        num_encoder_blocks=3,
        d_head_size=30,
        d_num_heads=3,
        d_ff_dim=50,
        num_decoder_blocks=3,
        dense_units=[120, 60, 30],
        e_dropout=0.2,
        d_dropout=0.2,
        dense_dropout=0.2
    )
    model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)
    stats, *_ = test_model(model, test_gen, dir, name, 'full_transformer')


def runTest6():
    X, Y, train_gen, val_gen, test_gen = prepare_data_full(lookback=48, lookforward=1, batch_size=48,
                                                           file='./raw_data/omni_5min_2010.csv')
    name = 'full_transformer_test'
    dir = './results/' + name
    encoder_shape = (X.shape[1], X.shape[2])
    decoder_shape = (1, Y.shape[1])
    model = full_transformer_model(
        encoder_shape,
        decoder_shape,
        e_head_size=30,
        e_num_heads=12,
        e_ff_dim=50,
        num_encoder_blocks=5,
        d_head_size=30,
        d_num_heads=12,
        d_ff_dim=50,
        num_decoder_blocks=5,
        dense_units=[120, 60, 30],
        e_dropout=0.2,
        d_dropout=0.2,
        dense_dropout=0.2
    )
    model, history, results = execute_model(model, train_gen, val_gen, test_gen, dir, name)
    stats, *_ = test_model(model, test_gen, dir, name, 'full_transformer')


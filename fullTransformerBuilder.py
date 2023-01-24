import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def transformer_decoder(inputs, encoder_inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, encoder_inputs)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    res = x + res
    return res


def full_transformer_model(encoder_shape, decoder_shape, e_head_size, e_num_heads, e_ff_dim, num_encoder_blocks,
                           d_head_size, d_num_heads, d_ff_dim, num_decoder_blocks, dense_units, e_dropout=0,
                           d_dropout=0, dense_dropout=0):
    encoder_inputs = layers.Input(shape=encoder_shape)
    x = encoder_inputs
    for _ in range(num_encoder_blocks):
        x = transformer_encoder(x, e_head_size, e_num_heads, e_ff_dim, e_dropout)
    encoder_res = layers.LayerNormalization(epsilon=1e-6)(x)

    decoder_inputs = layers.Input(shape=decoder_shape)
    x = decoder_inputs
    for _ in range(num_decoder_blocks):
        x = transformer_decoder(x, encoder_res, d_head_size, d_num_heads, d_ff_dim, d_dropout)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)

    for dim in dense_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(decoder_shape[1])(x)
    return Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

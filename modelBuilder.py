import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model





def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def transformer_model(input_shape, output_shape, head_size, num_heads, ff_dim, num_transformer_blocks, dense_units, dropout=0, dense_dropout=0):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D()(x)
    for dim in dense_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(output_shape)(x)
    return Model(inputs=inputs, outputs=outputs)

def transformer_model_f(input_shape, output_shape, head_size, num_heads, ff_dim, num_transformer_blocks, dense_units, dropout=0, dense_dropout=0):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.Flatten()(x)
    for dim in dense_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(output_shape)(x)
    return Model(inputs=inputs, outputs=outputs)

def classic_LSTM(input_shape, output_shape, lstm_nodes=20):
    inputs = layers.Input(shape=(input_shape[0], input_shape[1]))
    lstm1 = layers.LSTM(lstm_nodes, return_sequences=True)(inputs)
    lstm2 = layers.LSTM(lstm_nodes, return_sequences=True)(lstm1)
    lstm3 = layers.LSTM(lstm_nodes)(lstm2)
    output = layers.Dense(output_shape)(lstm3)
    model = Model(inputs=[inputs], outputs=output, name='Classic_LSTM')
    return model

# def singleInput(X, Y):
#     input_shape = (X.shape[1], X.shape[2])
#     inputs = layers.Input(shape=input_shape)
#     transformer_block0 = TransformerBlock(key_dim=11, num_heads=20, ff_dim=120)
#     x = transformer_block0(inputs)
#     for a in range(6):
#         transformer_block = TransformerBlock(key_dim=11, num_heads=20, ff_dim=120)
#         x = transformer_block(x)
#
#     x = layers.GlobalAveragePooling1D()(x)
#     x = layers.Dropout(0.1)(x)
#     x = layers.Dense(20, activation="relu")(x)
#     x = layers.Dense(40, activation="relu")(x)
#     x = layers.Dense(60, activation="relu")(x)
#
#     out = layers.Dense(50, activation="relu")(x)
#     output = layers.Dense(Y.shape[1])(out)
#     model = Model(inputs=inputs, outputs=output, name='single_data_test')
#     return model
#
#
# def mixedInputTest(X, Y):
#     X0, X1 = X
#     input_shape0 = (X0.shape[1], X0.shape[2])  # numerical data
#     input_shape1 = (X1.shape[1], X1.shape[2], X1.shape[3])  # images
#
#     # numerical data network
#     transformer_block0 = TransformerBlock(key_dim=11, num_heads=2, ff_dim=50)
#     inputs0 = layers.Input(shape=input_shape0)
#     x0 = transformer_block0(inputs0)
#     x0 = layers.GlobalAveragePooling1D()(x0)
#     x0 = layers.Dropout(0.1)(x0)
#     x0 = layers.Dense(50, activation="relu")(x0)
#     out0 = layers.Dropout(0.1)(x0)
#     model0 = Model(inputs=inputs0, outputs=out0)
#
#     # image data network
#     transformer_block1 = TransformerBlock(key_dim=512, num_heads=2, ff_dim=50)
#     inputs1 = layers.Input(shape=input_shape1)
#     x1 = transformer_block1(inputs1)
#     x1 = layers.GlobalAveragePooling2D(data_format='channels_first')(x1)
#     x1 = layers.Dropout(0.1)(x1)
#     x1 = layers.Dense(50, activation="relu")(x1)
#     out1 = layers.Dropout(0.1)(x1)
#     model1 = Model(inputs=inputs1, outputs=out1)
#
#     # combine the output
#     combine = layers.concatenate([model0.output, model1.output])
#
#     out = layers.Dense(50, activation="relu")(combine)
#     output = layers.Dense(Y.shape[1])(out)
#     model = Model(inputs=[model0.input, model1.input], outputs=output, name='mixed_data_test')
#     return model
#
#
# class TransformerBlock(layers.Layer):
#     def __init__(self, num_heads, key_dim, ff_dim, drop_rate=0.1):
#         super(TransformerBlock, self).__init__()
#         self.num_heads = num_heads
#         self.key_dim = key_dim
#         self.ff_dim = ff_dim
#         self.drop_rate = drop_rate
#         self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
#         self.ffn = keras.Sequential(
#             [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.key_dim), ]
#         )
#         self.layernorm0 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout0 = layers.Dropout(self.drop_rate)
#         self.dropout1 = layers.Dropout(self.drop_rate)
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "num_heads": self.num_heads,
#             "key_dim": self.key_dim,
#             "ff_dim": self.ff_dim,
#             "drop_rate": self.drop_rate,
#         })
#         return config
#
#     def call(self, inputs, training):
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout0(attn_output, training=training)
#         out0 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out0)
#         ffn_output = self.dropout1(ffn_output, training=training)
#         out1 = self.layernorm1(out0 + ffn_output)
#         return out1

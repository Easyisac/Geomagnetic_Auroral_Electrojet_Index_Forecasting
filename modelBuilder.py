import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def mixedInputTest(X, Y):
    X0, X1 = X
    input_shape0 = (X0.shape[1], X0.shape[2])  # numerical data
    input_shape1 = (X1.shape[1], X1.shape[2], X1.shape[3])  # images

    # numerical data network
    transformer_block0 = TransformerBlock(key_dim=11, num_heads=2, ff_dim=50)
    inputs0 = layers.Input(shape=input_shape0)
    x0 = transformer_block0(inputs0)
    x0 = layers.GlobalAveragePooling1D()(x0)
    x0 = layers.Dropout(0.1)(x0)
    x0 = layers.Dense(50, activation="relu")(x0)
    out0 = layers.Dropout(0.1)(x0)
    model0 = Model(inputs=inputs0, outputs=out0)

    # image data network
    transformer_block1 = TransformerBlock(key_dim=512, num_heads=2, ff_dim=50)
    inputs1 = layers.Input(shape=input_shape1)
    x1 = transformer_block1(inputs1)
    x1 = layers.GlobalAveragePooling2D(data_format='channels_first')(x1)
    x1 = layers.Dropout(0.1)(x1)
    x1 = layers.Dense(50, activation="relu")(x1)
    out1 = layers.Dropout(0.1)(x1)
    model1 = Model(inputs=inputs1, outputs=out1)

    # combine the output
    combine = layers.concatenate([model0.output, model1.output])

    out = layers.Dense(50, activation="relu")(combine)
    output = layers.Dense(Y.shape[1])(out)
    model = Model(inputs=[model0.input, model1.input], outputs=output, name='mixed_data_test')
    return model


class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, drop_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.drop_rate = drop_rate
        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.key_dim), ]
        )
        self.layernorm0 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout0 = layers.Dropout(self.drop_rate)
        self.dropout1 = layers.Dropout(self.drop_rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "drop_rate": self.drop_rate,
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout0(attn_output, training=training)
        out0 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out0)
        ffn_output = self.dropout1(ffn_output, training=training)
        out1 = self.layernorm1(out0 + ffn_output)
        return out1

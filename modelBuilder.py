import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

key_dim = 120
num_heads = 10
ff_dim = 10


def singleInputTransformer(x, y):
    input_shape = (x.shape[1], x.shape[2])
    inputs = layers.Input(shape=input_shape)
    transformer_block = TransformerBlock(key_dim=key_dim, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer_block(inputs)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(y.shape[1])(x)
    model = Model(inputs=inputs, outputs=output, name='single_Encoder')
    return model


class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, drop_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(key_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

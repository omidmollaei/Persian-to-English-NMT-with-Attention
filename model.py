"""Implement an encoder-decoder network using attention mechanism
to enhance its performance, building a neural machine translation
model to translate persian sentences to english."""

import tensorflow as tf
from tensorflow import keras

tokenizer_type = tf.keras.layers.TextVectorization


class Encoder(keras.layers.Layer):
    def __init__(self, units: int, tokenizer: tokenizer_type):
        super(Encoder, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocabulary_size()
        self.units = units

        self.embedding = keras.layers.Embedding(self.vocab_size, self.units, mask_zero=True)

        self.rnn = keras.layers.Bidirectional(
            merge_mode="sum",
            layer=keras.layers.GRU(
                self.units, return_sequences=True, recurrent_initializer="glorot_uniform"
            )
        )

    def call(self, inputs: tf.Tensor):
        vectors = self.embedding(inputs)
        outputs = self.rnn(vectors)
        return outputs

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.tokenizer(texts)
        context = self(context)
        return context


class CrossAttentionLayer(keras.layers.Layer):
    def __init__(self, units: int, num_heads: int, **kwargs):
        super().__init__()
        # self.last_attention_weights = None
        self.attention_layer = keras.layers.MultiHeadAttention(
            key_dim=units, num_heads=num_heads, **kwargs
        )
        self.layer_norm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()

    def call(self, x, context):
        """x is input to decode and context is encoder's output sequence"""
        attn_output, attn_scores = self.attention_layer(
            query=x,
            value=context,
            return_attention_scores=True)

        x = self.add([x, attn_output])
        x = self.layer_norm(x)
        return x



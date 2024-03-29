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
        self.attention_layer = keras.layers.MultiHeadAttention(
            key_dim=units, num_heads=num_heads, **kwargs
        )
        self.layer_norm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()

    def call(self, x, context):
        """x is input to decoder and context is encoder's output sequence"""
        attn_output, attn_scores = self.attention_layer(
            query=x,
            value=context,
            return_attention_scores=True)

        x = self.add([x, attn_output])
        x = self.layer_norm(x)
        return x


class Decoder(keras.layers.Layer):
    def __init__(self, units: int, attn_num_heads: int, tokenizer: tokenizer_type):
        super(Decoder, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocabulary_size()

        self.word_to_id = keras.layers.StringLookup(
            vocabulary=self.tokenizer.get_vocabulary(),
            mask_token="", oov_token="[UNK]")

        self.id_to_word = keras.layers.StringLookup(
            vocabulary=self.tokenizer.get_vocabulary(),
            mask_token="", oov_token="[UNK]", invert=True)

        self.start_token = self.word_to_id("[SOS]")
        self.end_token = self.word_to_id("[EOS]")
        self.units = units
        self.attn_num_heads = attn_num_heads

        self.embedding = keras.layers.Embedding(
            self.vocab_size, self.units, mask_zero=True)

        self.rnn = keras.layers.GRU(self.units, return_sequences=True,
                                    return_state=True, recurrent_initializer="glorot_uniform")

        self.attention = CrossAttentionLayer(units=self.units, num_heads=self.attn_num_heads)
        self.output_layer = keras.layers.Dense(self.vocab_size)  # produce logits

    def call(self, context, x, state=None, return_state=False):
        x = self.embedding(x)
        x, state = self.rnn(x, initial_state=state)
        x = self.attention(x, context)
        logits = self.output_layer(x)
        if return_state:
            return logits, state
        return logits

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(start_tokens)
        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):
        words = self.id_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        result = tf.strings.regex_replace(result, '^ *\[SOS\] *', '')
        result = tf.strings.regex_replace(result, ' *\[EOS\] *$', '')
        return result

    def get_next_token(self, context, next_token, done, state, temperature=0.0):
        logits, state = self(
            context, next_token,
            state=state, return_state=True)

        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :] / temperature
            next_token = tf.random.categorical(logits, num_samples=1)

        done = done | (next_token == self.end_token)
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)
        return next_token, done, state


class Translator(keras.models.Model):
    def __init__(self, inputs_tokenizer: tokenizer_type, targets_tokenizer: tokenizer_type,
                 units: int, attn_num_heads: int):
        super().__init__()

        self.encoder = Encoder(units, tokenizer=inputs_tokenizer)
        self.decoder = Decoder(units, attn_num_heads, tokenizer=targets_tokenizer)

    def translate(self, texts: int, *, max_length: int = 50, temperature: float = 0.0):
        # process the input texts
        context = self.encoder.convert_input(texts)
        batch_size = tf.shape(texts)[0]

        # set up the loop inputs
        tokens = []
        attention_weights = []
        next_token, done, state = self.decoder.get_initial_state(context)

        for _ in range(max_length):

            # generate the next token
            next_token, done, state = self.decoder.get_next_token(
                context, next_token, done, state, temperature)

            # collect the generated tokens
            tokens.append(next_token)
            # attention_weights.append(self.decoder.last_attention_weights)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        tokens = tf.concat(tokens, axis=-1)
        # self.last_attention_weights = tf.concat(attention_weights, axis=1)
        result = self.decoder.tokens_to_text(tokens)
        return result

    def call(self, x):
        context, x = x["enc_inputs"], x["dec_inputs"]
        context = self.encoder(context)
        logits = self.decoder(context, x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


def masked_loss(y_true, y_pred):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)

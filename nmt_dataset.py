"""
Load and prepare NMT dataset as a tf-dataset.
The dataset sources are :
    1) www.abadis.ir
    2) www.https://www.manythings.org/anki/
"""

import typing
import numpy as np
import tensorflow as tf
from dataclasses import field
from dataclasses import dataclass
from typing import Union, Callable

return_type = typing.Tuple[tf.data.Dataset,
                           typing.Tuple[tf.keras.layers.TextVectorization,
                                        tf.keras.layers.TextVectorization]]


@dataclass
class DatasetConfig:
    """A dataclass for holding parameters and config for loading dataset."""
    vocab_size: int = field(
        default=25_000,
        metadata={
            "help": "Total number of vocab size. Note that actual number of vocab size will be \
            two less than this number. The last two words are considered for start and end tokens."
        }
    )
    max_sequence_len: int = field(
        default=128,
        metadata={
            "help": "Maximum length of each sequence."
        }
    )
    batch_size: int = field(
        default=32,
        metadata={
            "help": "Dataset batch size when converting to tf-dataset."
        }
    )
    inputs_preprocessor: Union[None, Callable] = field(
        default=None,
        metadata={
            "help": "Input persian sentences preprocessor function. None for using default function."
        }
    )
    targets_preprocessor: Union[None, Callable] = field(
        default=None,
        metadata={
            "help": "Target english sentences preprocessor function. None for using default function."
        }
    )


# setup for persian preprocess function
fa_alphabets = ["آ", "ا", "ب", "پ", "ت", "ث", "ج", "چ", "ح", "خ",
                "د", "ذ", "ر", "ز", "ژ", "س", "ش", "ص", "ض", "ط",
                "ظ", "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن",
                "و", "ه", "ی", "ء", "1", "2", "3", "4", "5", "6",
                "7", "8", "9", "0", "؟"]
fa_reg_pattern = r"[^ \[\]a-zA-Z"
for a in fa_alphabets:
    fa_reg_pattern += a
fa_reg_pattern += "?!]+"


def build_preprocessor(func: Callable):
    def inner_preprocessor(*args, **kwargs):
        sentence = func(*args, **kwargs)

        # add start and end tokens
        sentence = tf.strings.join(['[SOS]', sentence, '[EOS]'], separator=' ')
        return sentence

    return inner_preprocessor


@build_preprocessor
def clean_fa(sentence: str) -> str:
    """Preprocess input persian sentence"""
    # create a space between a word and the punctuation following it.
    sentence = tf.strings.regex_replace(sentence, r"([<>?؟.!,])", r" \1 ")
    sentence = tf.strings.regex_replace(sentence, r'[" "]+', " ")
    sentence = tf.strings.regex_replace(sentence, fa_reg_pattern, '')
    sentence = tf.strings.strip(sentence)  # strip whitespace.
    return sentence


@build_preprocessor
def clean_en(sentence: str) -> str:
    """Preprocess target english sentence"""
    sentence = tf.strings.lower(sentence)
    sentence = tf.strings.regex_replace(sentence, '[^ a-z.?!,¿\[\]]', '')
    sentence = tf.strings.regex_replace(sentence, '[.?!,¿]', r' \0 ')
    sentence = tf.strings.strip(sentence)
    return sentence


def _load_data_from_disk():
    """Load source (persian) and target (english) sentences from files inside ./dataset directory.
    and these sentences as two list."""
    en_files = [f"./dataset/init_words/english_{i}.txt" for i in range(1, 7)]
    fa_files = [f"./dataset/init_words/persian_{i}.txt" for i in range(1, 7)]

    persian, english = list(), list()
    for eng_file, per_file in zip(en_files, fa_files):
        with open(eng_file, 'r') as f:
            english += f.readlines()
        with open(per_file, 'r', encoding="utf8") as f:
            persian += f.readlines()

    with open("./dataset/init_words/english_sentences.txt", 'r') as f:
        english += f.readlines()

    with open("./dataset/init_words/persian_sentences.txt", 'r', encoding="utf8") as f:
        persian += f.readlines()

    return persian, english


def load_dataset(config: DatasetConfig) -> return_type:
    """Load english to persian translation dataset.
    The data is from anki (https://www.manythings.org/anki/) and abadis (abadis.ir)."""
    persian, english = _load_data_from_disk()
    dataset = tf.data.Dataset.from_tensor_slices((persian, english))
    dataset = dataset.filter(lambda inputs, targets: tf.strings.length(inputs) > 0 and tf.strings.length(targets) > 0)
    en_sentences = dataset.map(lambda inputs, targets: targets)
    fa_sentences = dataset.map(lambda inputs, targets: inputs)

    inputs_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=config.vocab_size,
        standardize=build_preprocessor(config.inputs_preprocessor) if config.inputs_preprocessor else clean_fa,
        split="whitespace",
        output_sequence_length=config.max_sequence_len,
    )

    targets_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=config.vocab_size,
        standardize=build_preprocessor(config.targets_preprocessor) if config.targets_preprocessor else clean_en,
        split="whitespace",
        output_sequence_length=config.max_sequence_len,
    )

    inputs_vectorizer.adapt(fa_sentences)
    targets_vectorizer.adapt(en_sentences)

    dataset = dataset.map(lambda inputs, targets: (inputs_vectorizer(inputs), targets_vectorizer(targets)))
    dataset = dataset.map(lambda inputs, targets: (
        {"enc_inputs": inputs, "dec_inputs": targets[:-1]}, targets[1:]))
    dataset = dataset.cache()
    dataset = dataset.shuffle(len(english))
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, inputs_vectorizer, targets_vectorizer


def decode(sequence: Union[list, tf.Tensor, np.ndarray], vectorizer):
    seq_len = tf.reduce_sum(tf.cast(tf.math.not_equal(sequence, 0), dtype=tf.int32))
    seq = tf.gather(vectorizer.get_vocabulary(), sequence).numpy()
    words = [w.decode("utf8") if isinstance(w, bytes) else w for w in seq[:seq_len]]
    return " ".join(words)

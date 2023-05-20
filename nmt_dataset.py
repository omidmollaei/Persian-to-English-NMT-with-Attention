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
from typing import Union, Any, Callable

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

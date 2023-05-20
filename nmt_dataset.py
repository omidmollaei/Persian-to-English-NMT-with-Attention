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




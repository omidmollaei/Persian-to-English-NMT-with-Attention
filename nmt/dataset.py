"""Load and preprocess persian to english translation dataset."""

import os
import typing

import tensorflow as tf
import tensorflow_text as tf_text

default_path = os.path.join("./", "nmt", "data", "pes.txt")
return_type = typing.Tuple[tf.data.Dataset,
                           typing.Tuple[tf.keras.layers.TextVectorization,
                                        tf.keras.layers.TextVectorization]]

callable_type = typing.Callable[[str, str], typing.Tuple[typing.Union[str, None], str]]
post_processor_type = typing.Union[callable_type, None]


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def preprocess_persian_sentence(sentence: str) -> str:
    """Preprocess input persian sentence"""

    fa_alphabets = ["آ", "ا", "ب", "پ", "ت", "ث", "ج", "چ", "ح", "خ",
                    "د", "ذ", "ر", "ز", "ژ", "س", "ش", "ص", "ض", "ط",
                    "ظ", "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن",
                    "و", "ه", "ی", "ء", "1", "2", "3", "4", "5", "6",
                    "7", "8", "9", "0", "؟"]

    # add start and end tokens
    sentence = tf.strings.join(['[SOS]', sentence, '[EOS]'], separator=' ')

    # create a space between a word and the punctuation following it.
    sentence = tf.strings.regex_replace(sentence, r"([<>?؟.!,])", r" \1 ")
    sentence = tf.strings.regex_replace(sentence, r'[" "]+', " ")

    # replacing everything with space except (الف-ی, ".", "?", "!", ",", a-z, A-Z)
    reg_pattern = r"[^ \[\]a-zA-Z"
    for a in fa_alphabets:
        reg_pattern += a
    reg_pattern += "?.!]+"
    sentence = tf.strings.regex_replace(sentence, reg_pattern, '')
    sentence = tf.strings.strip(sentence)  # strip whitespace.

    return sentence


def preprocess_english_sentence(sentence: str) -> str:
    """preprocess target english sentence"""

    # split accented characters
    sentence = tf_text.normalize_utf8(sentence, "NFKD")
    sentence = tf.strings.lower(sentence)

    # keep space, a to z, and select punctuation.
    sentence = tf.strings.regex_replace(sentence, '[^ a-z.?!,¿\[\]]', '')

    # add spaces around punctuation.
    sentence = tf.strings.regex_replace(sentence, '[.?!,¿]', r' \0 ')

    # strip whitespace.
    sentence = tf.strings.strip(sentence)

    # add start and end tokens
    sentence = tf.strings.join(['[SOS]', sentence, '[EOS]'], separator=' ')

    return sentence


def load_data_from_disk(path):
    with open(path, "r", encoding="utf8") as file:
        lines = file.read()

    lines = lines.split("\n")
    return lines


def load_extend_dataset(extend_post_processor: post_processor_type = None) -> typing.Tuple[list, list]:
    english_sentences = load_data_from_disk(path=os.path.join("./", "nmt", "data", "english_sentences.txt"))
    persian_sentences = load_data_from_disk(path=os.path.join("./", "nmt", "data", "persian_sentences.txt"))
    clean_persian, clean_english = list(), list()
    for p, e in zip(persian_sentences, english_sentences):
        p, e = extend_post_processor(p, e)
        if p is not None:
            clean_persian.append(p)
            clean_english.append(e)

    return clean_persian, clean_english


def get_anki_dataset(vocab_size: int, batch_size: int,
                     path: str = default_path,
                     extend: bool = False,
                     extend_post_processor: post_processor_type = None) -> return_type:
    """prepare nmt dataset. The file names of additional data must be:
    english_sentences.txt and persian_sentences.txt and be placed inside nmt/data"""

    # load anki dataset
    lines = load_data_from_disk(path=path)
    inputs, targets = [], []
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        inputs.append(parts[1])    # persian sentence
        targets.append(parts[0])   # english sentence

    # load additional dataset
    if extend:
        persian_sentences, english_sentences = load_extend_dataset(extend_post_processor)
        inputs = inputs + persian_sentences
        targets = targets + english_sentences

    # build and initialize vectorizer layers
    inputs_text_processor = tf.keras.layers.TextVectorization(
        standardize=preprocess_persian_sentence,
        max_tokens=vocab_size,
        ragged=True)
    inputs_text_processor.adapt(inputs)

    targets_text_processor = tf.keras.layers.TextVectorization(
        standardize=preprocess_english_sentence,
        max_tokens=vocab_size,
        ragged=True)
    targets_text_processor.adapt(targets)

    # build tf-dataset
    inputs = inputs_text_processor(inputs).to_tensor()       # zero padding
    targets = targets_text_processor(targets).to_tensor()    # zero padding
    dataset = tf.data.Dataset.from_tensor_slices((
        {"enc_inputs": inputs, "dec_inputs": targets[:, :-1]}, targets[:, 1:]
    ))
    dataset = dataset.cache()
    dataset = dataset.shuffle(len(inputs))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, (inputs_text_processor, targets_text_processor)

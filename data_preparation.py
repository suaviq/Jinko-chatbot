from cgi import test
import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
use_builtins = True

#----------------------- PREPARE DATA-------------------------------
def load_data(path_to_file):
    file = open(path_to_file,'r').read()
    qna_list = [f.split('\t') for f in file.split('\n')]
    x = [x[0] for x in qna_list]
    y = [y[1] for y in qna_list]
    return x, y

# create a tf.data dataset
def tf_dataset(x, y):
    BUFFER_SIZE = len(x)
    BATCH_SIZE = 64
    TRAIN_SIZE = int(0.8 * BUFFER_SIZE/BATCH_SIZE)
    TEST_SIZE = int(0.2 * BUFFER_SIZE/BATCH_SIZE)

    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_data = dataset.take(TRAIN_SIZE)
    test_data = dataset.skip(TRAIN_SIZE)
    setup_data = dataset.take(1)
    return train_data, test_data, setup_data

#----------------------- TEXT PREPROCESSING-------------------------------
def unicode_normalization(text):
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text
#----------------------- TEXT VECTORIZATION-------------------------------
def text_vectorization(x, y):
    max_vocab_size = 5000
    input_text_processor = tf.keras.layers.TextVectorization(standardize=unicode_normalization, max_tokens=max_vocab_size)
    input_text_processor.adapt(x)
    output_text_processor = tf.keras.layers.TextVectorization(standardize=unicode_normalization, max_tokens=max_vocab_size)
    output_text_processor.adapt(y)
    input_vocab = np.array(input_text_processor.get_vocabulary())
    return input_text_processor, output_text_processor


# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def _int64_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     num_seq):

    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    for _ in range(num_seq):
        yield np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()

sequences = random_sequences(length_from=3, length_to=8,
                                   vocab_lower=3, vocab_upper=10,
                                   num_seq=3000)
sos = 1
eos = 2

writer = tf.python_io.TFRecordWriter('file.tfrecord')

for seq in sequences:
    feature = {
            'encoder_inputs' : _int64_feature(seq),
            'decoder_inputs' : _int64_feature([sos]+seq),
            'decoder_targets' : _int64_feature(seq+[eos])
            }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()
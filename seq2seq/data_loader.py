import tensorflow as tf
import os
from data_tfrecord import random_sequences

def train_input_fn(base_dir, batch_size=32, num_epochs=None):

    def _parse_function(example_proto):
        keys_to_features = {
            'encoder_inputs': tf.VarLenFeature(dtype=tf.int64),
            'decoder_inputs': tf.VarLenFeature(dtype=tf.int64),
            'decoder_targets': tf.VarLenFeature(dtype=tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        encoder_inputs = tf.sparse_tensor_to_dense(parsed_features['encoder_inputs'])
        decoder_inputs = tf.sparse_tensor_to_dense(parsed_features['decoder_inputs'])
        decoder_targets = tf.sparse_tensor_to_dense(parsed_features['decoder_targets'])
        encoder_inputs = tf.cast(encoder_inputs, tf.int32)
        decoder_inputs = tf.cast(decoder_inputs, tf.int32)
        decoder_targets = tf.cast(decoder_targets, tf.int32)
        return encoder_inputs, decoder_inputs, decoder_targets

    filenames = tf.gfile.Glob(os.path.join(base_dir, '*.tfrecord'))
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None], [None]))
    iterator = dataset.make_one_shot_iterator()
    encoder_inputs, decoder_inputs, decoder_targets = iterator.get_next()

    return {'encoder_inputs': encoder_inputs, 'decoder_inputs': decoder_inputs, 
            'decoder_targets': decoder_targets}


def test_input_fn(num):
    def sequences():
        return random_sequences(length_from=3, length_to=8,
                                   vocab_lower=3, vocab_upper=10,
                                   num_seq=num)
    dataset = tf.data.Dataset.from_generator(sequences,tf.int32)
    dataset = dataset.padded_batch(2,padded_shapes=([None]))
    iterator = dataset.make_one_shot_iterator()
    encoder_inputs = iterator.get_next()
    return {'encoder_inputs': encoder_inputs}





if __name__ == '__main__':
    train_dir = './'
    feature = test_input_fn(10)
    with tf.Session() as sess:
        print(sess.run(feature))

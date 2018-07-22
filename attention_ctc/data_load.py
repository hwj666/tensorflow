# -*- coding: utf-8 -*-

import tensorflow as tf
import os

def get_train_input(base_dir, batch_size=32, num_epochs=None):
    def dense_to_sparse(dense_tensor):
        indices = tf.where(tf.not_equal(dense_tensor, 0))
        values = tf.gather_nd(dense_tensor, indices)
        shape = tf.shape(dense_tensor, out_type=tf.int64)
        return tf.SparseTensor(indices, values, shape)

    def _parse_function(example_serialized):
        feature_map = {
            'image':  tf.FixedLenFeature( [], dtype=tf.string),
            'label':  tf.VarLenFeature( dtype=tf.int64), 
            'input':  tf.VarLenFeature( dtype=tf.int64),
            'target': tf.VarLenFeature( dtype=tf.int64)
        }
        features = tf.parse_single_example( example_serialized, feature_map )
    
        image = tf.image.decode_jpeg( features['image'], channels=1 )
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.subtract(image, 0.5)
        label = tf.sparse_tensor_to_dense(features['label'])
        encode_input = tf.sparse_tensor_to_dense(features['input'])
        encode_target = tf.sparse_tensor_to_dense(features['target'])
 
        return image,label,encode_input,encode_target
    
    filenames = tf.gfile.Glob(os.path.join(base_dir,'*.tfrecord'))
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.padded_batch(batch_size,padded_shapes=([None,None,1],[None],[None],[None]))
    iterator = dataset.make_one_shot_iterator() 
    image,label,encode_input,encode_target = iterator.get_next()
    label = dense_to_sparse(label)
    label = tf.cast(label,tf.int32)
    return {'image':image, 'label':label, 'input':encode_input, 'target':encode_target}

def get_test_inputs(base_dir):

    def _prase(filename):
        image = tf.read_file(filename)
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.subtract(image,0.5)
        return image,
    filenames = tf.gfile.Glob(base_dir+'*.png')
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(_prase)
    dataset = dataset.batch(1)
    element = dataset.make_one_shot_iterator()
    image = element.get_next()
    
    return  {'image':image}


if __name__ == '__main__':
    features = get_train_input('../tfdata/svt/',batch_size=32)
    label = features['input']
    with tf.Session() as sess:
        ffff = sess.run(label)
        print(len(ffff))

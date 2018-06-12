# -*- coding: utf-8 -*-

import tensorflow as tf
import os

out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

def num_classes():
    return len(out_charset)

def get_train_input(base_dir, batch_size=32, num_epochs=None):
    def dense_to_sparse(dense_tensor):
        indices = tf.where(tf.not_equal(dense_tensor, 0))
        values = tf.gather_nd(dense_tensor, indices)
        shape = tf.shape(dense_tensor, out_type=tf.int64)
        return tf.SparseTensor(indices, values, shape)


    def _parse_function(example_serialized):
        feature_map = {
            'image/encoded':  tf.FixedLenFeature( [], dtype=tf.string, 
                                                  default_value='' ),
            'image/labels':   tf.VarLenFeature( dtype=tf.int64 ), 
            'image/width':    tf.FixedLenFeature( [], dtype=tf.int64,
                                                  default_value=1 ),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                                 default_value='' ),
            'text/string':     tf.FixedLenFeature([], dtype=tf.string,
                                                 default_value='' ),
            'text/length':    tf.FixedLenFeature( [], dtype=tf.int64,
                                                  default_value=1 )
        }
        features = tf.parse_single_example( example_serialized, feature_map )
    
        image = tf.image.decode_jpeg( features['image/encoded'], channels=1 )
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.subtract(image, 0.5)
        first_row = tf.slice(image, [0, 0, 0], [1, -1, -1]) # -1 is all first row,all col,all channel
        image = tf.concat([first_row, image], 0)
        width = tf.cast( features['image/width'], tf.int32) # for ctc_loss
        label = tf.sparse_tensor_to_dense(features['image/labels']) # for batching
        
        length = features['text/length']
        text = features['text/string']
        filename = features['image/filename']
        return image,width,label,length,text,filename
    
    
    
    filenames = tf.gfile.Glob(os.path.join(base_dir,'*.tfrecord'))
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.padded_batch(batch_size,padded_shapes=([None,None,1],[],[None],[],[],[]))
    iterator = dataset.make_one_shot_iterator() 
    image,width,label,length,text,filename = iterator.get_next()
    label = dense_to_sparse(label)
    label = tf.cast(label,tf.int32)
    return {'image':image,'width':width,'label':label,'length':length,'text':text,'filename':filename}

def get_test_inputs(base_dir):

    def _prase(filename):
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.subtract(image,0.5)
        return image,tf.shape(image)[1]
    filenames = tf.gfile.Glob(base_dir+'*.jpg')
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(_prase)
    dataset = dataset.batch(1)
    element = dataset.make_one_shot_iterator()
    image,width = element.get_next()
    
    return  {'image':image,'width':width,'label':None}



if __name__ == '__main__':
#    features = get_train_input('../data/train/')
#    with tf.Session() as sess:
#        label = features['label']
#
#        print(label.eval())
    feautes = get_test_inputs('../data/test/')
    with tf.Session() as sess:
        print(sess.run(feautes['width']))
        
        
        
        
        

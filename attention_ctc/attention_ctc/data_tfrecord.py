# coding=utf-8
import os
import tensorflow as tf
import math
import numpy as np

data_path = '../data/svt1/'

charset_dir = data_path + 'charset.txt'
anno_dir = data_path + 'anno.txt'

with open(charset_dir,encoding='utf-8') as f:
    charset = f.read()
out_charset = list(charset)

out_charset = ['PAD','SOS','EOS'] + out_charset

char_to_int = dict(zip(out_charset, range(len(out_charset))))
int_to_char = dict(zip(char_to_int.values(), char_to_int.keys()))

num_classes = len(out_charset)

jpeg_data = tf.placeholder(dtype=tf.string)
jpeg_decoder = tf.image.decode_jpeg(jpeg_data,channels=3)

kernel_sizes = [3,3,3,3,3,3,3,3] # CNN kernels for image reduction

# Minimum allowable width of image after CNN processing
min_width = 20

def calc_seq_len(image_width):

    conv1_trim =  2 * (kernel_sizes[0] // 2)
    
    after_conv1 = image_width - conv1_trim 
    after_pool2 = after_conv1 // 2
    after_pool4 = after_pool2 - 1
    after_pool6 =  after_pool4 - 1
    after_pool8 =  after_pool6
    seq_len = after_pool8
    return seq_len

seq_lens = [calc_seq_len(w) for w in range(1024)]

def gen_data(label_path, output_filebase, num_shards=1):

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    sess = tf.Session(config=session_config)
    with open(label_path,encoding='utf-8') as lines:
        image_infos = lines.readlines()
    
    num_digits = math.ceil( math.log10( num_shards + 1 ))
    shard_format = '%0'+ ('%d'%num_digits) + 'd'

    batch_range = np.linspace(0,len(image_infos),num_shards+1).astype(np.int)
    for i in range(num_shards):
        start,end = batch_range[i],batch_range[i+1]
        out_filename = output_filebase+'-'+(shard_format % i)+'.tfrecord'
        print('{} of {} [{}:{}]{}'.format(i,num_shards,start,end,out_filename))
        gen_shard(sess, image_infos[start:end], out_filename)
    sess.close()

def gen_shard(sess, img_info, output_filename):
    """Create a TFRecord file from a list of image filenames"""
    writer = tf.python_io.TFRecordWriter(output_filename)
    for info in img_info:
        path, label = info.strip().split(' ')
        path = data_path + path
        try:
            image_data, width = get_image(sess,path)
            label_code = [char_to_int[c] for c in label]
            
            if is_writable(width,label_code):
                example = make_example(image_data, label_code)
                writer.write(example.SerializeToString())
            else:
                print('SKIPPING', path)
        except:
            # Some files have bogus payloads, catch and note the error, moving on
            print('ERROR',path)
    writer.close()


def get_image(sess,filename):

    """Given path to an image file, load its data and size"""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()
    image = sess.run(jpeg_decoder, feed_dict={jpeg_data: image_data})

    width = image.shape[1]
    return image_data, width

def is_writable(image_width, text):
    """Determine whether the CNN-processed image is longer than the string"""
    return (image_width > min_width) and (len(text) <= seq_lens[image_width])

def make_example(image_data, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(tf.compat.as_bytes(image_data)),
        'label': _int64_feature(label),
        'input' : _int64_feature([char_to_int['SOS']]+label),
        'target' : _int64_feature(label+[char_to_int['SOS']])
    }))
    return example

def _int64_feature(values):
    if not isinstance(values,list):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def main(argv=None):
    gen_data(anno_dir, '../tfdata/svt/words')

if __name__ == '__main__':
    main()





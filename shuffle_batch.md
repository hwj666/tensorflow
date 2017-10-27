超级简述版


```
import os
import tensorflow as tf

from skimage.io import imread
import numpy as np

data_path = 'uu'
tf_data_path = 'uu/tf_data/'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_data(train_data_path):
    tfrecords_filename = 'a.tfrecords'
    writer = tf.python_io.TFRecordWriter(os.path.join(tf_data_path, tfrecords_filename))
    img_name = os.listdir(train_data_path)
    for image in img_name:
        if '.bmp' in image:
            img = imread(os.path.join(train_data_path, image), as_grey=True)  # 读取数据
            image_raw = img.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
    writer.close()


def read_data(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    return image


def main():
    tfrecords_filename = ['uu/tf_data/a.tfrecords']
    filename_queue = tf.train.string_input_producer(tfrecords_filename)
    img = read_data(filename_queue)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image = sess.run(img)
        image = image.astype(np.uint8)
        print(image.shape)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # create_data(data_path)
    main()

```
(6102656,)

我这个图片是1112*686=762832，但是输出的却是762832 * 8 = 6102656，也就是说是输出是期望的8倍，这是因为使用imread读取数据默认是float64的，也就是8个字节，
而我们最后输出的图片要求是一个字节的，这就相差了8倍
要不然，如果直接reshape(1112,686)会报错
> tensorflow.python.framework.errors_impl.InvalidArgumentError: Input to reshape is a tensor with 6102656 values, but the requested shape has 762832

因为
> as_grey : bool
If True, convert color images to grey-scale (64-bit floats). Images that are already in grey-scale format are not converted.

我也不知道如何转换，所以就变成cv2了，opencv默认读取的时候是直接读取三个通道的，哪怕是灰度图，而且数值类型为uint8。



---

简单版


```
import os
import tensorflow as tf
import cv2

data_path = 'dataset/uu/'
tf_data_path = 'dataset/uu/tf_data/'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_data(train_data_path):
    img_name = [img for img in os.listdir(train_data_path) if '.bmp' in img]
    for i in range(5):
        tfrecords_filename = str(i) + '.tfrecords'
        writer = tf.python_io.TFRecordWriter(os.path.join(tf_data_path, tfrecords_filename))
        for j in range(10):
            for image in img_name:
                img = cv2.imread(os.path.join(train_data_path, image))
                height, width = img.shape[0], img.shape[1]
                image_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'image_raw': _bytes_feature(image_raw)
                }))
                writer.write(example.SerializeToString())
        writer.close()


def read_data(data_path):
    reader = tf.TFRecordReader()
    files = tf.train.match_filenames_once(os.path.join(data_path, '*.tfrecords'))
    #读取数据的时候忘了添加路径，只读了文件名，一直报错，请求10个数据，但是只有0个数据，让我检查了好久
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, (height, width, -1))
    return image


def main():

    img = read_data(tf_data_path)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image = sess.run(img)
        cv2.imshow('img', image)

        cv2.waitKey()
        cv2.destroyAllWindows()

        print(image)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    create_data(data_path)
    main()


```
这个会将所有的数据都返回来，我们训练的时候其实是想要batch训练的，因此需要改一下，例如加上


```
    batch_size = 10
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    img_batch = tf.train.shuffle_batch([image], batch_size=batch_size, capacity=capacity,
                                       min_after_dequeue=min_after_dequeue)
```
但是就会报错
> ValueError: All shapes must be fully defined: [TensorShape([Dimension(None), Dimension(None), Dimension(None)])]

这是因为shuffle_batch的输入必须都是尺寸已知的，，然而我们的输入image并没有明确的定义尺寸，如果你的图片大小是已知的，你可以直接在reshape的时候

```
image = tf.reshape(image, (420, 580, 3))
```
但通常不是，因此我们需要resize

```
image = tf.reshape(image, (height, width, 3))#先还原图片
image = tf.image.resize_images(image, (420, 580))#接着缩放
```
但是你会发现画出的图还是很奇怪，这是因为==tf.image.resize_images==函数返回的是float类型，我们将其转换成uint8就可以了


```
image = tf.reshape(image, (height, width, 3))#先还原图片
image = tf.image.resize_images(image, (420, 580))#接着缩放
image = tf.cast(image, tf.uint8)#转成可视化的类型，如果下面还需要进行float类型的处理，可以不用转化，这里只是为了显示
```


复杂版
```
import os
import tensorflow as tf
import cv2

data_path = 'dataset/uu/'
tf_data_path = 'dataset/uu/tf_data/'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_data(train_data_path):
    img_name = [img for img in os.listdir(train_data_path) if '.bmp' in img]
    for i in range(5):
        tfrecords_filename = str(i) + '.tfrecords'
        writer = tf.python_io.TFRecordWriter(os.path.join(tf_data_path, tfrecords_filename))
        for j in range(10):
            for image in img_name:
                img = cv2.imread(os.path.join(train_data_path, image))
                height, width = img.shape[0], img.shape[1]
                image_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'image_raw': _bytes_feature(image_raw)
                }))
                writer.write(example.SerializeToString())
        writer.close()


def read_data(data_path):
    reader = tf.TFRecordReader()
    files = tf.train.match_filenames_once(os.path.join(data_path, '*.tfrecords'))
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, (height, width, 3))

    image = tf.image.resize_images(image, (420, 580))
    image = tf.cast(image, tf.uint8)
    batch_size = 10
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    img_batch = tf.train.shuffle_batch([image], batch_size=batch_size, capacity=capacity,
                                       min_after_dequeue=min_after_dequeue)
    return img_batch


def main():
    img = read_data(tf_data_path)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image = sess.run(img)
        cv2.imshow('img', image[0])

        cv2.waitKey()
        cv2.destroyAllWindows()

        print(image)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    create_data(data_path)
    main()

```

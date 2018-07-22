import tensorflow as tf
import data_load as dl
from model import crnn_model

def main(unused_argv):
    train_dir = '../tfdata/svt/'
    test_dir = '../data/test/'
    crnn = tf.estimator.Estimator(model_fn=crnn_model,model_dir='model')
    crnn.train(input_fn=lambda: dl.get_train_input(train_dir))   
    predictions = crnn.predict(input_fn=lambda:dl.get_test_inputs(test_dir))

    for pre in predictions:
        label = pre['classes']
        print(list(map(lambda x:dl.out_charset[x],label)))
if __name__ == '__main__':
    tf.app.run()
    
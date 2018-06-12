# -*- coding: utf-8 -*-
import tensorflow as tf

import model
import mjsynth


def _get_training(rnn_logits,label,sequence_length):
    """Set up training ops"""
    with tf.name_scope("train"):

        loss = model.ctc_loss_layer(rnn_logits,label,sequence_length)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):

            learning_rate = tf.train.exponential_decay(
                learning_rate=1e-4,
                global_step=tf.train.get_global_step(),
                decay_steps=2**16,
                decay_rate=0.9,
                name='learning_rate')

            optimizer = tf.train.AdamOptimizer(learning_rate)
            
            train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())

            tf.summary.scalar( 'learning_rate', learning_rate )
            tf.summary.scalar( 'loss', loss )
    return train_op,loss



def crnn_model(features, mode):

    image,width,label = features['image'],features['width'],features['label']

    feature,sequence_length = model.convnet_layers( image, width, mode)
    logits = model.rnn_layers(feature, sequence_length, mjsynth.num_classes())
    decoded,log_probabilities = tf.nn.ctc_beam_search_decoder(logits,sequence_length,beam_width=128)
    predictions = {
            'classes': tf.sparse_tensor_to_dense(decoded[0]),
            'prob' : log_probabilities
            }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    train_op, loss = _get_training(logits, label, sequence_length)
    log_dict = {
            'loss':loss,
            }
    logging_hook = tf.train.LoggingTensorHook(log_dict,every_n_iter=1)
    summary_hook = tf.train.SummarySaverHook(save_steps=1,output_dir='log/',summary_op=tf.summary.merge_all())
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,training_hooks = [logging_hook,summary_hook])
    
    
    

    
def main(unused_argv):
    train_dir = '../data/train/'
    test_dir = '../data/test/'
    crnn = tf.estimator.Estimator(model_fn=crnn_model,model_dir='model')
#    crnn.train(input_fn=lambda: mjsynth.get_train_input(train_dir))    
    predictions = crnn.predict(input_fn=lambda:mjsynth.get_test_inputs(test_dir))

    for pre in predictions:
        label = pre['classes']
        
        print(list(map(lambda x:mjsynth.out_charset[x],label)))
if __name__ == '__main__':
    tf.app.run()
    
    
    
    
    
    
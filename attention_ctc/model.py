import tensorflow as tf
import ctc_model as cm
import data_tfrecord as dtf

def crnn_model(features, mode):
    
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    image = features['image']
    
    feature,sequence_length = cm.convnet_layers(image, training)
    ctc_logits = cm.rnn_layers(feature, dtf.num_classes)

    if mode == tf.estimator.ModeKeys.TRAIN:
        label = features['label']
        with tf.name_scope("train"):
            loss = cm.ctc_loss_layer(ctc_logits, label, sequence_length)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                learning_rate = tf.train.exponential_decay(
                    learning_rate=1e-4,
                    global_step=tf.train.get_global_step(),
                    decay_steps=2**16,
                    decay_rate=0.9,
                    name='learning_rate')

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                gvs = optimizer.compute_gradients(loss)
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(capped_gvs)

                train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())

                tf.summary.scalar( 'learning_rate', learning_rate )
                tf.summary.scalar( 'loss', loss )

        log_dict = {'loss':loss}
        logging_hook = tf.train.LoggingTensorHook(log_dict,every_n_iter=1)
        summary_hook = tf.train.SummarySaverHook(save_steps=1,output_dir='log/',summary_op=tf.summary.merge_all())
        
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,training_hooks = [logging_hook,summary_hook])

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = tf.transpose(ctc_logits, (1,0,2))
        decoded,log_probabilities = tf.nn.ctc_beam_search_decoder(logits,sequence_length,beam_width=3)
        predictions = {
            'classes': tf.sparse_tensor_to_dense(decoded[0]),
            'prob' : log_probabilities
            }
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

def model_fn(features, mode, params):
    encoder_inputs = features['encoder_inputs']
    encoder_shape = tf.shape(encoder_inputs)
    embedding = tf.get_variable("embedding_encoder", [params.vocab_size, params.embedding_size])
    encoder_emb_inputs = tf.nn.embedding_lookup(embedding, encoder_inputs)
    encoder_cell = tf.nn.rnn_cell.LSTMCell(params.num_units)
    decoder_cell = tf.nn.rnn_cell.LSTMCell(params.num_units)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inputs, dtype=tf.float32)
    projection_layer = layers_core.Dense(params.vocab_size, use_bias=False)
    
    if params.use_attention:
        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            params.num_units, encoder_outputs,
            memory_sequence_length=None)

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=params.num_units)

        initial_state = decoder_cell.zero_state(encoder_shape[0], tf.float32).clone(cell_state=encoder_state)
    else:
        initial_state = encoder_state   


    if mode == tf.estimator.ModeKeys.PREDICT:
        
        if params.use_beamsearch:
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                    initial_state, multiplier=params.beam_width)

            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=embedding,
                    start_tokens=tf.fill([encoder_shape[0]], params.sos_id),
                    end_token=params.eos_id,
                    initial_state=decoder_initial_state,
                    beam_width=params.beam_width,
                    output_layer=projection_layer,
                    length_penalty_weight=0.0)
        else:
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding,
                tf.fill([encoder_shape[0]], params.sos_id), params.eos_id)
            # Inference Decoder
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, inference_helper, initial_state,
                output_layer=projection_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder, maximum_iterations=params.maximum_iterations)
        translations = outputs.predicted_ids if params.use_beamsearch else outputs.sample_id

        predictions = {
            'translations': translations
            }

        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)



    decoder_inputs = features['decoder_inputs']
    decoder_targets = features['decoder_targets']
    decoder_shape = tf.shape(decoder_targets)
    decoder_lengths = tf.fill([decoder_shape[0]],decoder_shape[1])
    decoder_emb_inputs = tf.nn.embedding_lookup(embedding, decoder_inputs)
    train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, decoder_lengths)
    train_decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, train_helper, initial_state, output_layer=projection_layer)
    final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)
    logits = tf.identity(final_outputs.rnn_output, 'logits')
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=decoder_targets, logits=logits)
    loss = tf.reduce_mean(loss)
    
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        learning_rate = tf.train.exponential_decay(
            learning_rate=1e-4,
            global_step=tf.train.get_global_step(),
            decay_steps=2**16,
            decay_rate=0.9,
            name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs,global_step=tf.train.get_global_step())

    tf.summary.scalar( 'learning_rate', learning_rate )
    tf.summary.scalar('loss',loss)



    log_dict = {'loss':loss}
    
    logging_hook = tf.train.LoggingTensorHook(log_dict,every_n_iter=1)
    summary_hook = tf.train.SummarySaverHook(save_steps=1,output_dir='log/',summary_op=tf.summary.merge_all())
 
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,training_hooks = [logging_hook,summary_hook])
import tensorflow as tf
from model import model_fn
from data_loader import train_input_fn,test_input_fn

tf.reset_default_graph()

hparams = tf.contrib.training.HParams(
    sos_id = 1,
    eos_id = 2,
    num_units=6,
    vocab_size=10,
    embedding_size=8,
    learning_rate = 0.01,
    maximum_iterations = 20,
    use_beamsearch = False,
    beam_width =3,
    use_attention = False,
)


def main(unused_argv):
    train_dir = './'
    seq2seq = tf.estimator.Estimator(model_fn=model_fn,params=hparams,model_dir='model')
    seq2seq.train(input_fn=lambda: train_input_fn(train_dir,num_epochs=50))
    predictions = seq2seq.predict(input_fn=lambda:test_input_fn(6))
    for pre in predictions:
        print(pre['translations'])

if __name__ == '__main__':
    tf.app.run()

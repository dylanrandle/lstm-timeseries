''' 
Define Tensorflow graph for LSTM model
'''

import tensorflow as tf
from tensorflow.contrib.rnn import rnn

class Model():
    def __init__(self, args):
        inputs = tf.placeholder(tf.float32, [args.num_batches, args.batch_size, 1])
        lstm = rnn.BasicLSTMCell(args.hidden_dim)
        hidden_state = tf.zeros([args.batch_size, lstm.state_size])
        current_state = tf.zeros([args.batch_size, lstm.state_size])
        state = hidden_state, current_state


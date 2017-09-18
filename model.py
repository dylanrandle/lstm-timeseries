''' 
Define Tensorflow graph for LSTM model
'''
import tensorflow as tf
import numpy as np

class Model():
    def __init__(self, args):
        self.input = tf.placeholder('float', [args.batch_size, args.seq_len, 1])
        self.target = tf.placeholder('float', [args.batch_size, 1])

        self.weights = tf.Variable(tf.random_normal([args.hidden_dim, 1]))
        self.biases = tf.Variable(tf.random_normal([1]))

        x = tf.reshape(self.input, [-1, args.seq_len])
        x = tf.split(x, args.seq_len, 1)

        self.cell = tf.contrib.rnn.BasicLSTMCell(args.hidden_dim)
        self.outputs, self.states = tf.contrib.rnn.static_rnn(self.cell, x, dtype=tf.float32)

        # we only care about the last output for predicting price
        self.pred = tf.matmul(self.outputs[-1], self.weights) + self.biases

        # calculate loss
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.pred, self.target))
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(self.loss)

        # save loss/pred for tensorboard
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('pred', self.pred[0][0])


'''
Train the model
'''

import tensorflow as tf
import argparse
import utils
import numpy as np
import matplotlib.pyplot as plt
# from model import Model

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', type=str, default='data/TQQQ.csv',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--seq_len', type=int, default=50,
                        help='number of days to look back')
    parser.add_argument('--hidden_dim', type=int, default=6,
                        help='number of hidden units')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for optimization')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of passes through training data')
    args = parser.parse_args()
    train(args)

def train(args):
    X_train, X_test, Y_train, Y_test = utils.read_timeseries(args.data_file)

    inputs = tf.placeholder(tf.float32, shape=[None, None, 1])
    targets = tf.placeholder(tf.float32, shape=[None, 1])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(args.hidden_dim)
    batch_size = tf.shape(inputs)[1]
    initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=initial_state, time_major=True)

    W = tf.Variable(tf.zeros([args.hidden_dim, 1]))
    b = tf.Variable(tf.zeros([1]))
    logits = tf.matmul(rnn_outputs[-1], W) + b

    loss_op = tf.losses.mean_squared_error(targets, logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for e in range(args.num_epochs):
            ## TODO: X_train/Y_train need to properly match inputs/targets dimensions
            sess.run(train_op, feed_dict={inputs: X_train, targets: Y_train})
            print('ran a session')

if __name__ == '__main__':
    main()
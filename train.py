'''
Train the model
'''
import os
import time
import tensorflow as tf
import argparse
import utils
import numpy as np
import matplotlib.pyplot as plt
from model import Model

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
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for optimization')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of passes through training data')
    args = parser.parse_args()
    train(args)

def train(args):
    X_train, X_test, Y_train, Y_test = utils.read_timeseries(args.data_file)
    num_batches = int(len(X_train)/args.seq_len)
    model = Model(args)

    with tf.Session() as sess:
        # tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

        for e in range(args.num_epochs):
            print('Epoch %s/%s' % (e+1,args.num_epochs))

            for i in range(num_batches):
                X_slice = X_train[i*args.seq_len:(i+1)*args.seq_len].reshape((1, args.seq_len, 1))
                Y_target = Y_train[i+1].reshape((1, 1))
                summ, train_loss, _ = sess.run([summaries, model.loss_op, model.train_op], feed_dict={model.inputs: X_slice, model.targets: Y_target})
                writer.add_summary(summ, e*num_batches + i)

if __name__ == '__main__':
    main()
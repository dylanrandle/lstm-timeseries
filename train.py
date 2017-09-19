'''
Train the model
'''
import os
import time
import tensorflow as tf
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from six.moves import cPickle
from model import Model

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', type=str, default='data/train.csv',
                        help='training data file (as csv)')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--save_every', type=int, default=5000,
                        help='save frequency')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--seq_len', type=int, default=50,
                        help='length of historical data fed to LSTM')
    parser.add_argument('--hidden_dim', type=int, default=6,
                        help='LSTM hidden dimension')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of batches to execute')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for SGD optimizer')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='number of passes through training data')
    args = parser.parse_args()
    train(args)

def train(args):    
    X = np.asarray(pd.read_csv(args.data_file, usecols=['Adj Close']))
    # X = normalize(X, norm='max')
    num_iters = len(X) - args.seq_len

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'args.pkl'), 'wb') as f:
        cPickle.dump(args, f)

    model = Model(args)
    with tf.Session() as sess:
        # tensorboard & saver
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        for e in range(args.num_epochs):
            for b in range(num_iters):
                x = X[b:b+args.seq_len].reshape((1, args.seq_len, 1))
                y = X[b+args.seq_len].reshape((1, 1))
                summ, train_loss, train_op, pred = sess.run([summaries, model.loss, model.train_op, model.pred], feed_dict={model.input: x, model.target: y})
                writer.add_summary(summ, e*num_iters + b)
                if b % 10 == 0:
                    print('Epoch: %s/%s | Iteration: %s/%s | Loss: %s | Pred: %s | True: %s' % (e+1, args.num_epochs, b, num_iters, train_loss, pred, X[b+args.seq_len]))

            # checkpoint the model after every epoch
            checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=e*num_iters + b)

if __name__ == '__main__':
    main()
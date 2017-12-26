'''
Calculate and plot predictions on test data
'''
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import cPickle
from model import Model
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_file', type=str, default='data/sp500_test.csv',
                        help='test data file (as csv)')
    parser.add_argument('--train_file', type=str, default='data/sp500_train.csv',
                        help='training data file (as csv)')
    parser.add_argument('--load_from', type=str, default='save',
                        help='directory to load model and config from')
    args = parser.parse_args()
    test(args)

def test(args):
    X_train = np.asarray(pd.read_csv(args.train_file, usecols=['Adj Close']))
    norm_factor = np.amax(X_train)
    X = np.asarray(pd.read_csv(args.test_file, usecols=['Adj Close']))
    X = X / norm_factor
    # load saved args
    with open(os.path.join(args.load_from, 'args.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)

    model = Model(saved_args)
    preds = []
    actuals = []
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.load_from)
        if ckpt and ckpt.model_checkpoint_path:
            # load saved model
            saver.restore(sess, ckpt.model_checkpoint_path)
        num_iters = len(X) - saved_args.seq_len
        for b in range(num_iters):
            x = X[b:b+saved_args.seq_len].reshape((1, saved_args.seq_len, 1))
            y = X[b+saved_args.seq_len].reshape((1, 1))
            pred = sess.run([model.pred], feed_dict={model.input: x, model.target: y})
            pred = pred[0][0][0]
            actual = y[0][0]
            preds.append(pred*norm_factor)
            actuals.append(actual*norm_factor)
    plt.plot(preds)
    plt.plot(actuals)
    plt.show()

if __name__ == '__main__':
    main()
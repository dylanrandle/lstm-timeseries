'''
Calculate and plot predictions on test data
'''
import os
import argparse
import tensorflow as tf
from six.moves import cPickle
from model import Model

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', type=str, default='data/test.csv',
                        help='training data file (as csv)')
    parser.add_argument('--load_from', type=str, default='save',
                        help='directory to load model and config from')
    args = parser.parse_args()
    test(args)

def test(args):
    X = np.asarray(pd.read_csv(args.data_file, usecols=['Adj Close']))

    # load saved args
    with open(os.path.join(args.load_from, 'args.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)

    model = Model(saved_args)
    preds = []
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
            y = Y[b+saved_args.seq_len].reshape((1, 1))
            train_loss, pred = sess.run([model.loss, model.pred], feed_dict={model.input: x, model.target: y})
    print(preds)

if __name__ == '__main__':
    main()
'''
Calculate and plot predictions on test data
'''
import argparse
import utils
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', type=str, default='data/test.csv',
                        help='training data file (as csv)')

def test(args, model, X, Y, X_prev):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ## TODO: need to calculate state before starting predictions
        ## TODO: change how data is loaded/passed into model
        for i in range(len(X)):
            x = np.append(X[i], X_prev[i-50:])
            x = x.reshape((1, 50, 1))
            print(x)
            pred = model.predict(sess, x)
            print(pred)

    ## seed with first data point
    print('Not implemented.')

    ## plot prediction and actual
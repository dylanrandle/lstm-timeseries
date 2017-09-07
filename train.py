'''
Train the model
'''

import keras
import tensorflow as tf
import argparse

from model import Model

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/TQQQ.csv',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=6,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='gru or lstm')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=1,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    # parser.add_argument('--save_every', type=int, default=1000,
    #                     help='save frequency')
    # parser.add_argument('--grad_clip', type=float, default=5.,
    #                     help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    # parser.add_argument('--decay_rate', type=float, default=0.97,
    #                     help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')
    # parser.add_argument('--init_from', type=str, default=None,
    #                     help="""continue training from saved model at this path. Path must contain files saved by previous training process:
    #                         'config.pkl'        : configuration;
    #                         'chars_vocab.pkl'   : vocabulary definitions;
    #                         'checkpoint'        : paths to model file(s) (created by tf).
    #                                               Note: this file contains absolute paths, be careful when moving files around;
    #                         'model.ckpt-*'      : file(s) with model definition (created by tf)
    #                     """)
    args = parser.parse_args()
    train(args)

def train(args):
    ''' Using Keras '''
    # model = keras.models.Sequential()
    # if args.model == 'lstm':
    #     model.add(keras.layers.recurrent.LSTM(args.rnn_size, input_shape=(args.batch_size, args.seq_length, 1)))
    # elif args.model == 'gru':
    #     model.add(keras.layers.recurrent.GRU(args.rnn_size, input_shape=(args.batch_size, args.seq_length, 1)))
    # model.add(keras.layers.core.Dense(1))
    # print('Model summary: ')
    # model.summary()

if __name__ == '__main__':
    main()
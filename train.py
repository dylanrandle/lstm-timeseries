'''
Train the model
'''

import tensorflow as tf
from tensorflow.contrib.rnn import rnn
import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/TQQQ.csv',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    args = parser.parse_args()
    train(args)

def train(args, timesteps=50, hidden_units=6):
    cell = rnn.BasicLSTMCell(hidden_units)

if __name__ == '__main__':
    main()
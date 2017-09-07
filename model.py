'''
Build the model 
'''

import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

class Model():
    def __init__(self, args, training=True):
        self.args = args

        if args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        else:
            raise Exception('Model type not supported: {}'.format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell, input_keep_prob=args.input_keep_prob
                                                output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        if len(cells) > 1:
            self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
        else:
            self.cell = cell = cells

        



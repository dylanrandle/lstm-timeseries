''' 
Define Tensorflow graph for LSTM model
'''

import tensorflow as tf

class Model():
    def __init__(self, args):
        self.inputs = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.targets = tf.placeholder(tf.float32, shape=[None, 1])

        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(args.hidden_dim)
        batch_size = tf.shape(self.inputs)[1]
        self.initial_state = self.lstm_cell.zero_state(batch_size, tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(self.lstm_cell, self.inputs, initial_state=self.initial_state, time_major=True)
        self.last_state = rnn_states

        W = tf.Variable(tf.zeros([args.hidden_dim, 1]))
        b = tf.Variable(tf.zeros([1]))
        self.logits = tf.matmul(rnn_outputs[-1], W) + b
        
        self.loss_op = tf.losses.mean_squared_error(self.targets, self.logits)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)

        tf.summary.scalar('train loss', self.loss_op)

    def predict(self, sess, x, prime):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        
        ## prime the net with the past 50 days
        for day in prime:
            

        pred = sess.run([self.logits], feed_dict={self.inputs: x})
        return pred


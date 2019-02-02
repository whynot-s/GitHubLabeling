import tensorflow as tf
import numpy as np


class TextRNN:

    def __init__(self, batch_size, embedding_size, context_size, classes_num, correct_threshold, sequence_length):
        # Inputs
        self.input_X = tf.placeholder(tf.float32, [batch_size, sequence_length, embedding_size])
        self.input_Y = tf.placeholder(tf.float32, [batch_size, classes_num])
        self.x1 = tf.transpose(self.input_X, [1,0,2])
        # Initial States
        self.c_left = tf.placeholder(tf.float32, [batch_size, context_size])
        self.c_right = tf.placeholder(tf.float32, [batch_size, context_size])
        # Hidden Layer to Hidden Layer
        self.W_left = tf.Variable(tf.random_normal([context_size, context_size]))
        self.W_right =  tf.Variable(tf.random_normal([context_size, context_size]))
        # To Combine Embedding
        self.Ws_left = tf.Variable(tf.random_normal([embedding_size, context_size]))
        self.Ws_right = tf.Variable(tf.random_normal([embedding_size, context_size]))

        current_state_left = self.c_left
        current_state_right = self.c_right

        state_series_left = []
        state_series_right = []

        x_unstacked = tf.unstack(self.input_X, axis=1)
        x_reversed_unstacked = tf.unstack(tf.reverse(self.input_X, [0]), axis=1)

        for current_input in x_unstacked:
            next_state_left = tf.math.add(
                tf.matmul(current_state_left, self.W_left),
                tf.matmul(current_input, self.Ws_left))
            state_series_left.append(next_state_left)
            current_state_left = next_state_left

        for current_input in x_reversed_unstacked:
            next_state_right = tf.math.add(
                tf.matmul(current_state_right, self.W_right),
                tf.matmul(current_input, self.Ws_right))
            state_series_right.append(next_state_right)
            current_state_right = next_state_right

        ssr_reversed = tf.reverse(state_series_right, [0])
        concated = tf.expand_dims(tf.transpose(tf.concat([state_series_left, self.x1, ssr_reversed], 2), [1, 0, 2]), 3)
        pooled = tf.nn.max_pool(tf.tanh(concated), [1, 3, 1, 1], [1, 1, 1, 1], padding='VALID')

        logits = tf.contrib.layers.fully_connected(tf.reshape(pooled, [batch_size, -1]), classes_num, activation_fn=tf.sigmoid)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_Y, logits=logits)
        self.loss = tf.reduce_mean(losses)

        pred = tf.nn.sigmoid(logits)
        label_true = tf.count_nonzero(self.input_Y)
        logit_temp = tf.where(pred >= correct_threshold, x=pred, y=tf.zeros(tf.shape(pred)))
        logit_true = tf.count_nonzero(logit_temp)
        pred_correct = tf.count_nonzero(tf.multiply(tf.cast(self.input_Y, tf.float32), logit_temp))

        self.precision = tf.divide(tf.cast(pred_correct, tf.float32), tf.cast(logit_true, tf.float32))
        self.recall = tf.divide(tf.cast(pred_correct, tf.float32), tf.cast(label_true, tf.float32))


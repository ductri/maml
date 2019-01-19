import numpy as np
import tensorflow as tf


class SineModel:
    def __init__(self, input_dim, output_dim):
        self.meta_weights = dict(
            w1=tf.get_variable(name='w1', dtype=tf.float32, shape=(input_dim, 40)),
            b1=tf.get_variable(name='b1', dtype=tf.float32, shape=40),
            w2=tf.get_variable(name='w2', dtype=tf.float32, shape=(40, output_dim)),
            b2=tf.get_variable(name='b2', dtype=tf.float32, shape=1)
        )

    def infer(self, tf_X, weights):
        tf_logits = tf.matmul(tf_X, weights['w1']) + weights['b1']
        tf_logits = tf.nn.relu(tf_logits)
        tf_logits = tf.matmul(tf_logits, weights['w2']) + weights['b2']
        return tf_logits

    def get_loss(self, tf_logits, tf_y):
        tf_loss = tf.losses.mean_squared_error(labels=tf_y, predictions=tf_logits)
        return tf_loss

    def do_it(self, tf_X, tf_y, weights):
        tf_logits = self.infer(tf_X, weights)
        tf_loss = self.get_loss(tf_logits, tf_y)
        return tf_logits, tf_loss

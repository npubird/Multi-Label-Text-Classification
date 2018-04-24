# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import tensorflow as tf


def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        W = tf.get_variable("W", [output_size, input_size], dtype=input_.dtype)
        b = tf.get_variable("b", [output_size], dtype=input_.dtype)

    return tf.nn.xw_plus_b(input_, tf.transpose(W), b)


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope=('highway_lin_{0}'.format(idx))))
            t = tf.sigmoid(linear(input_, size, scope=('highway_gate_{0}'.format(idx))) + bias)
            output = t * g + (1. - t) * input_
            input_ = output

    return output


class TextHAN(object):
    """A HAN for text classification."""

    def __init__(
            self, sequence_length, num_classes, top_num, batch_size, vocab_size, hidden_size,
            embedding_size, embedding_type, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool)

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained by our corpus
            if pretrained_embedding is None:
                self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                             name="embedding")
            else:
                if embedding_type == 0:
                    self.embedding = tf.constant(pretrained_embedding, name="embedding")
                    self.embedding = tf.cast(self.embedding, tf.float32)
                if embedding_type == 1:
                    self.embedding = tf.Variable(pretrained_embedding, name="embedding", trainable=True)
                    self.embedding = tf.cast(self.embedding, tf.float32)
            self.embedded_sentence = tf.nn.embedding_lookup(self.embedding, self.input_x)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[hidden_size, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), dtype=tf.float32, name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.embedded_sentence, W, b, name="logits")
            self.scores = tf.sigmoid(self.logits, name="scores")
            self.topKPreds = tf.nn.top_k(self.scores, k=top_num, sorted=True, name="topKPreds")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            self.loss = tf.reduce_mean(losses, name="loss") + l2_reg_lambda * l2_loss

# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow import sigmoid
from tensorflow import tanh
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm


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


class BatchNormLSTMCell(rnn.RNNCell):
    """Batch normalized LSTM (cf. http://arxiv.org/abs/1603.09025)"""

    def __init__(self, num_units, is_training=False, forget_bias=1.0,
                 activation=tanh, reuse=None):
        """Initialize the BNLSTM cell.

        Args:
          num_units: int, The number of units in the BNLSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
        """
        self._num_units = num_units
        self._is_training = is_training
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=self._reuse):
            c, h = state
            input_size = inputs.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                                   [input_size, 4 * self._num_units],
                                   initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                                   [self._num_units, 4 * self._num_units],
                                   initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self._num_units])

            xh = tf.matmul(inputs, W_xh)
            hh = tf.matmul(h, W_hh)

            bn_xh = batch_norm(xh, self._is_training)
            bn_hh = batch_norm(hh, self._is_training)

            hidden = bn_xh + bn_hh + bias

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=hidden, num_or_size_splits=4, axis=1)

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
            bn_new_c = batch_norm(new_c, 'c', self._is_training)
            new_h = self._activation(bn_new_c) * sigmoid(o)
            new_state = rnn.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)


def bn_lstm_identity_initializer(scale):

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        """
        Ugly cause LSTM params calculated in one matrix multiply
        """

        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = orthogonal([size, size])
        t[:, size * 2:size * 3] = orthogonal([size, size])
        t[:, size * 3:] = orthogonal([size, size])
        return tf.constant(t, dtype=dtype)

    return _initializer


def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer


class TextRNN(object):
    """A RNN for text classification."""

    def __init__(
            self, sequence_length, num_classes, vocab_size, hidden_size, fc_hidden_size,
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

        with tf.name_scope("Bi-lstm"):
            # Bi-LSTM Layer
            lstm_fw_cell = rnn.BasicLSTMCell(hidden_size)  # forward direction cell
            lstm_bw_cell = rnn.BasicLSTMCell(hidden_size)  # backward direction cell
            if self.dropout_keep_prob is not None:
                lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

            # Creates a dynamic bidirectional recurrent neural network
            # shape: [batch_size, sequence_length, hidden_size]
            outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                             self.embedded_sentence, dtype=tf.float32)

        # Concat output
        self.lstm_concat = tf.concat(outputs, axis=2)  # [batch_size, sequence_length, hidden_size*2]
        self.lstm_out = tf.reduce_mean(self.lstm_concat, axis=1)  # [batch_size, hidden_size*2]

        # Fully Connected Layer
        with tf.name_scope("fc"):
            W = tf.Variable(tf.truncated_normal(shape=[hidden_size*2, fc_hidden_size], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[fc_hidden_size]), dtype=tf.float32, name="b")
            self.fc = tf.nn.xw_plus_b(self.lstm_out, W, b)

            # Batch Normalization Layer
            self.fc_bn = batch_norm(self.fc, is_training=self.is_training)

            # Apply nonlinearity
            self.fc_out = tf.nn.relu(self.fc_bn, name="relu")

        # Highway Layer
        self.highway = highway(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=0, scope="Highway")

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.highway, W, b, name="logits")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import tensorflow as tf
import copy


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


def get_context_left(context_left, embedding_previous, W_l, W_sl):
    """
    :param context_left: [batch_size, embedding_size]
    :param embedding_previous: [batch_size, embedding_size]
    :return: output: [None, embedding_size]
    """
    left_c = tf.matmul(context_left, W_l)
    left_e = tf.matmul(embedding_previous, W_sl)
    left_h = left_c + left_e
    context_left = tf.nn.tanh(left_h)
    return context_left


def get_context_right(context_right, embedding_afterward, W_r, W_sr):
    """
    :param context_right: [batch_size, embedding_size]
    :param embedding_afterward: [batch_size, embedding_size]
    :return: output: [None,embed_size]
    """
    right_c = tf.matmul(context_right, W_r)
    right_e = tf.matmul(embedding_afterward, W_sr)
    right_h = right_c + right_e
    context_right = tf.nn.tanh(right_h)
    return context_right


class TextRCNN(object):
    """A RCNN for text classification."""

    def __init__(
            self, sequence_length, num_classes, batch_size, vocab_size, hidden_size,
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
            self.embedded_sentence = tf.nn.embedding_lookup(self.embedding, self.input_x)  # [None, sentence_length, embedding_size]

        # get splitted list of word embeddings
        self.embedded_word_split = tf.split(self.embedded_sentence, sequence_length, axis=1)  # sentence_length 个 [None, 1, embedding_size]
        self.embedded_word_squeezed = [tf.squeeze(x, axis=1) for x in self.embedded_word_split]   # sentence_length 个 [None, embedding_size]

        # Get list of context left
        embedding_previous = tf.Variable(tf.truncated_normal(shape=[batch_size, embedding_size], stddev=0.1),
                                         name="left_side_first_word")
        context_left_previous = tf.zeros((batch_size, embedding_size))

        context_left_list = []
        W_l = tf.Variable(tf.truncated_normal(shape=[embedding_size, embedding_size], stddev=0.1), name="W_l")
        W_sl = tf.Variable(tf.truncated_normal(shape=[embedding_size, embedding_size], stddev=0.1), name="W_sl")
        for i, current_embedding_word in enumerate(self.embedded_word_squeezed):
            context_left = get_context_left(context_left_previous, embedding_previous, W_l, W_sl)  # [None, embedding_size]
            context_left_list.append(context_left)
            embedding_previous = current_embedding_word
            context_left_previous = context_left

        # ---------------------------------------------------------------------------

        # Get copy of list of reversed context word embeddings for next step
        embedded_chars_squeezed_reverse = copy.copy(self.embedded_word_squeezed)
        embedded_chars_squeezed_reverse.reverse()

        # Get list of context right
        embedding_afterward = tf.Variable(tf.truncated_normal(shape=[batch_size, embedding_size], stddev=0.1),
                                          name="right_side_last_word")
        context_right_afterward = tf.zeros((batch_size, embedding_size))

        context_right_list = []
        W_r = tf.Variable(tf.truncated_normal(shape=[embedding_size, embedding_size], stddev=0.1), name="W_r")
        W_sr = tf.Variable(tf.truncated_normal(shape=[embedding_size, embedding_size], stddev=0.1), name="W_sr")
        for j, current_embedding_word in enumerate(self.embedded_word_squeezed):
            context_right = get_context_right(context_right_afterward, embedding_afterward, W_r, W_sr)  # [None, embdedding_size]
            context_right_list.append(context_right)
            embedding_afterward = current_embedding_word
            context_right_afterward = context_right

        # ---------------------------------------------------------------------------

        # Ensemble (left, embedding, right) to output
        output_list = []
        for index, current_embedding_word in enumerate(self.embedded_word_squeezed):
            representation = tf.concat([context_left_list[index], current_embedding_word, context_right_list[index]],
                                       axis=1)
            output_list.append(representation)  # sentence_length 个 [None, embedding_size*3]

        # Stack list to a tensor
        self.output_rnn = tf.stack(output_list, axis=1)

        # Maxpooling over the outputs
        self.output_pooling = tf.reduce_max(self.output_rnn, axis=1)  # [None, embedding_size*3]

        # Highway Layer
        self.highway = highway(self.output_pooling, self.output_pooling.get_shape()[1],
                               num_layers=1, bias=0, scope="Highway")

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[hidden_size*3, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

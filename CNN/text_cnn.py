# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


def linear(input_, output_size, name=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    :param input_: a tensor or a list of 2D, batch x n, Tensors.
    :param output_size: int, second dimension of W[i].
    :param name: VariableScope for the created subgraph; defaults to "Linear".
    :returns: A 2D Tensor with shape [batch x output_size] equal to \
                sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    :raises: ValueError, if some of the arguments has unspecified or wrong shape.
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: {}".format(str(shape)))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: {}".format(str(shape)))
    input_size = shape[1]

    # Now the computation.
    with tf.name_scope(name or "SimpleLinear"):
        W = tf.Variable(tf.truncated_normal(shape=[output_size, input_size], stddev=0.1), dtype=input_.dtype, name="W")
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), dtype=input_.dtype, name="b")

    return tf.nn.xw_plus_b(input_, W, b)


def highway(input_, size, num_layers=1, bias=-2.0, act=tf.nn.relu, name=None):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.name_scope(name or "Highway"):
        for idx in range(num_layers):
            g = act(linear(input_, size, name='highway_lin_{}'.format(idx)))
            t = tf.sigmoid(linear(input_, size, name='highway_gate_{}'.format(idx)) + bias)
            output = t * g + (1. - t) * input_
            input_ = output

    return output


class TextCNN(object):
    """A CNN for text classification."""

    def __init__(
            self, sequence_length, num_classes, vocab_size, fc_hidden_size, embedding_size,
            embedding_type, filter_sizes, num_filters, l2_reg_lambda=0.0, pretrained_embedding=None):

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
            # 默认采用的是随机生成正态分布的词向量。
            # 也可以是通过自己的语料库训练而得到的词向量。
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
            self.embedded_sentence_expanded = tf.expand_dims(self.embedded_sentence, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-filter{}".format(filter_size)):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), dtype=tf.float32, name="b")
                conv = tf.nn.conv2d(
                    self.embedded_sentence_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Batch Normalization Layer
                conv_bn = batch_norm(tf.nn.bias_add(conv, b), is_training=self.is_training)

                # Apply nonlinearity
                conv_out = tf.nn.relu(conv_bn, name="relu")

            with tf.name_scope("pool-filter{}".format(filter_size)):
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    conv_out,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")

            pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.pool = tf.concat(pooled_outputs, 3)
        self.pool_flat = tf.reshape(self.pool, [-1, num_filters_total])

        # Fully Connected Layer
        with tf.name_scope("fc"):
            W = tf.Variable(tf.truncated_normal(shape=[num_filters_total, fc_hidden_size], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[fc_hidden_size]), dtype=tf.float32, name="b")
            self.fc = tf.nn.xw_plus_b(self.pool_flat, W, b)

            # Batch Normalization Layer
            self.fc_bn = batch_norm(self.fc, is_training=self.is_training)

            # Apply nonlinearity
            self.fc_out = tf.nn.relu(self.fc_bn, name="relu")

        # Highway Layer
        # with tf.name_scope("highway"):
        #    self.highway = highway(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=0, name="Highway")

        # Add dropout
        with tf.name_scope("dropout"):
            self.fc_drop = tf.nn.dropout(self.fc_out, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), dtype=tf.float32, name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.fc_drop, W, b, name="logits")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

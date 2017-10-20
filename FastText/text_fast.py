# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


class TextFAST(object):
    """A FASTTEXT for text classification."""

    def __init__(
            self, sequence_length, num_classes, vocab_size, fc_hidden_size, embedding_size,
            embedding_type, l2_reg_lambda=0.0, pretrained_embedding=None):

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

        # Average Vectors
        self.embedded_sentence_average = tf.reduce_mean(self.embedded_sentence, axis=1) # [batch_size, embedding_size]

        # Fully Connected Layer
        with tf.name_scope("fc"):
            W = tf.Variable(tf.truncated_normal(shape=[embedding_size, fc_hidden_size], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[fc_hidden_size]), dtype=tf.float32, name="b")
            self.fc = tf.nn.xw_plus_b(self.embedded_sentence_average, W, b)

            # Batch Normalization Layer
            self.fc_bn = batch_norm(self.fc, is_training=self.is_training)

            # Apply nonlinearity
            self.fc_out = tf.nn.relu(self.fc_bn, name="relu")

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

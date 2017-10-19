# -*- coding:utf-8 -*-

import tensorflow as tf


class TextHAN(object):
    """
    A FASTTEXT for text classification.
    Uses an embedding layer, followed by a bi-lstm, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, batch_size, vocab_size, hidden_size,
            embedding_size, embedding_type, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="global_Step")
        self.initializer = tf.random_normal_initializer(stddev=0.1)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # 默认采用的是随机生成正态分布的词向量。
            # 也可以是通过自己的语料库训练而得到的词向量。
            if pretrained_embedding is None:
                self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            else:
                if embedding_type == 0:
                    self.W = tf.constant(pretrained_embedding, name="W")
                    self.W = tf.cast(self.W, tf.float32)
                if embedding_type == 1:
                    self.W = tf.Variable(pretrained_embedding, name="W", trainable=True)
                    self.W = tf.cast(self.W, tf.float32)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  # [None, sentence_length, embedding_size]



        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[hidden_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.embedded_chars_average, W, b, name="logits")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

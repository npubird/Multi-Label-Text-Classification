# -*- coding:utf-8 -*-

import tensorflow as tf
import copy


class TextRCNN(object):
    """
    A RCNN for text classification.
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
            self.embedded_chars_split = tf.split(self.embedded_chars, sequence_length, axis=1)  # sentence_length 个 [None, 1, embedding_size]
            self.embedded_chars_squeezed = [tf.squeeze(x, axis=1) for x in self.embedded_chars_split]   # sentence_length 个 [None, embedding_size]

        # Get list of context left
        self.W_l = tf.get_variable("W_l", shape=[embedding_size, embedding_size], initializer=self.initializer)
        self.W_sl = tf.get_variable("W_sl", shape=[embedding_size, embedding_size], initializer=self.initializer)

        embedding_previous = tf.get_variable("left_side_first_char", shape=[batch_size, embedding_size],
                                             initializer=self.initializer)
        context_left_previous = tf.zeros((batch_size, embedding_size))
        context_left_list = []
        for i, current_embedding_char in enumerate(self.embedded_chars_squeezed):
            context_left = self.get_context_left(context_left_previous, embedding_previous)  # [None,embedding_size]
            context_left_list.append(context_left)  # append result to list
            embedding_previous = current_embedding_char  # assign embedding_previous
            context_left_previous = context_left  # assign context_left_previous

        # Get list of context right
        self.W_r = tf.get_variable("W_r", shape=[embedding_size, embedding_size], initializer=self.initializer)
        self.W_sr = tf.get_variable("W_sr", shape=[embedding_size, embedding_size], initializer=self.initializer)

        self.embedded_chars_squeezed_reverse = copy.copy(self.embedded_chars_squeezed)
        self.embedded_chars_squeezed_reverse.reverse()
        embedding_afterward = tf.get_variable("right_side_last_char", shape=[batch_size, embedding_size],
                                              initializer=self.initializer)
        context_right_afterward = tf.zeros((batch_size, embedding_size))
        context_right_list = []
        for j, current_embedding_char in enumerate(self.embedded_chars_squeezed_reverse):
            context_right = self.get_context_right(context_right_afterward, embedding_afterward)
            context_right_list.append(context_right)
            embedding_afterward = current_embedding_char
            context_right_afterward = context_right

        # Ensemble (left, embedding, right) to output
        output_list = []
        for index, current_embedding_char in enumerate(self.embedded_chars_squeezed):
            representation = tf.concat([context_left_list[index], current_embedding_char, context_right_list[index]],
                                       axis=1)
            output_list.append(representation)  # sentence_length 个 [None, embedding_size*3]

        # Stack list to a tensor
        self.output_rnn = tf.stack(output_list, axis=1)


        # Maxpooling over the outputs
        self.output_pooling = tf.reduce_max(self.output_rnn, axis=1)  # [None, embedding_size*3]

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output_pooling, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[hidden_size*3, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

    def get_context_left(self, context_left, embedding_previous):
        """
        :param context_left:
        :param embedding_previous:
        :return: output:[None, embedding_size]
        """
        left_c = tf.matmul(context_left, self.W_l)  # context_left:[batch_size, embedding_size];W_l:[embedding_size, embedding_size]
        left_e = tf.matmul(embedding_previous, self.W_sl)   # embedding_previous;[batch_size, embedding_size]
        left_h = left_c + left_e
        context_left = tf.nn.tanh(left_h)
        return context_left

    def get_context_right(self, context_right, embedding_afterward):
        """
        :param context_right:
        :param embedding_afterward:
        :return: output:[None,embed_size]
        """
        right_c = tf.matmul(context_right, self.W_r)  # context_right:[batch_size, embedding_size];W_r:[embedding_size, embedding_size]
        right_e = tf.matmul(embedding_afterward, self.W_sr)   # embedding_afterward;[batch_size, embedding_size]
        right_h = right_c + right_e
        context_right = tf.nn.tanh(right_h)
        return context_right

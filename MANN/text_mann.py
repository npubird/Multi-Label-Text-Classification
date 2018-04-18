import os
import time
import sys
import datetime
import itertools
import numpy as np
import tensorflow as tf
from collections import Counter
from gensim.models import Word2Vec
import json
import random
import gensim
from Data_Process import *
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import gflags

def lrelu(x, leak=1e-2):
    """Leak relu activation.

    :param x: Input tensor.
    :param leak: Leak. Default is 1e-2.
    :return: Output tensor.
    """
    return tf.maximum(x, leak * x)

class LSTMnet(object):
    def __init__(self, hidden_neurons,keep_prob):
        self.hidden_neurons = hidden_neurons
        self.keep_prob = keep_prob # dropout keep prob
        # create rnn cell

        lstm_layer_left = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_neurons,state_is_tuple=True)
        self.cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_layer_left,output_keep_prob=self.keep_prob)

        # hidden_layers_lefts = []
        # for idx, hidden_size in enumerate(self.hidden_neurons):
        #     lstm_layer_left = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        #     hidden_layer_left = tf.contrib.rnn.DropoutWrapper(cell=lstm_layer_left,
        #                                                  output_keep_prob=self.keep_prob)
        #     hidden_layers_lefts.append(hidden_layer_left)
        # self.cell = tf.contrib.rnn.MultiRNNCell(cells=hidden_layers_lefts, state_is_tuple=True)

    def state_output(self,input_data,sequence_len):
        state_series,current_state = tf.nn.dynamic_rnn(cell=self.cell,
                                                                       inputs=input_data,
                                                                       sequence_length=sequence_len,
                                                                       dtype=tf.float32)
        return state_series,current_state


class MANN(object):
    def __init__(self, config,pretrained_embedding = None,embedding_type = 0):
        self.input_size = input_size = config["input_size"]
        self.vocab_size = config["vocab_size"]
        self.image_emb_size = config["image_emb_size"]
        self.domain_emb_size = config["domain_emb_size"]
        self.keep_prob_value = config["keep_prob"]
        self.max_images_num = config['max_images_num']
        self.max_domains_num = config['max_domains_num']
        self.max_text_len = config['max_text_len']
        self.all_domains_num = config['all_domains_num']
        self.margin = config['margin']
        self.score_layer_size1 = config['score_layer_size1']
        self.l2_reg = config['l2_reg']
        self.datapath = config['datapath']


        self.soft_attention_size = 100

        self.LSTM_hidden_neurons = self.input_size + self.domain_emb_size + self.image_emb_size

        qid_im_embed = self.qid_images = tf.placeholder(tf.float32,
                                         [None, self.max_images_num, self.domain_emb_size])
        posqid_im_embed = self.posqid_images = tf.placeholder(tf.float32,
                                            [None, self.max_images_num, self.domain_emb_size])
        negqid_im_embed = self.negqid_images = tf.placeholder(tf.float32,
                                            [None, self.max_images_num, self.domain_emb_size])

        self.qid_images_mask = tf.placeholder(tf.float32, [None, self.max_images_num])
        self.posqid_images_mask = tf.placeholder(tf.float32, [None, self.max_images_num])
        self.negqid_images_mask = tf.placeholder(tf.float32, [None, self.max_images_num])

        # self.images = tf.placeholder(tf.float32, [3,None,self.max_images_num,self.image_height,self.image_width])
        self.domains = tf.placeholder(tf.int32, [3,None,self.max_domains_num])
        # self.qid_images_mask = tf.placeholder(tf.float32, [None, self.max_images_num])
        # self.posqid_images_mask = tf.placeholder(tf.float32, [None, self.max_images_num])
        # self.negqid_images_mask = tf.placeholder(tf.float32, [None, self.max_images_num])
        self.qid_domains_mask = tf.placeholder(tf.float32, [None, self.max_domains_num])
        self.posqid_domains_mask = tf.placeholder(tf.float32, [None, self.max_domains_num])
        self.negqid_domains_mask = tf.placeholder(tf.float32, [None, self.max_domains_num])



        # self.max_steps = tf.placeholder(tf.int32)  # max seq length of current batch

        self.qid_input = tf.placeholder(tf.int32, [None, self.max_text_len])
        self.posqid_input = tf.placeholder(tf.int32, [None, self.max_text_len])
        self.negqid_input = tf.placeholder(tf.int32, [None, self.max_text_len])
        # self.qid_length = tf.placeholder(tf.int32, [None])
        # self.posqid_length = tf.placeholder(tf.int32, [None])
        # self.negqid_length = tf.placeholder(tf.int32, [None])

        self.qid_mask = tf.placeholder(tf.float32, [None,self.max_text_len])
        self.posqid_mask = tf.placeholder(tf.float32, [None,self.max_text_len])
        self.negqid_mask = tf.placeholder(tf.float32, [None,self.max_text_len])

        self.qid_mask_last = tf.placeholder(tf.float32, [None, self.max_text_len])
        self.posqid_mask_last = tf.placeholder(tf.float32, [None, self.max_text_len])
        self.negqid_mask_last = tf.placeholder(tf.float32, [None, self.max_text_len])

        self.keep_prob = tf.placeholder(tf.float32)  # dropout keep prob

        # self.label = tf.placeholder(tf.float32, [None])

        length_shape = tf.shape(self.qid_mask)
        batch_size = length_shape[0]

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # 默认采用的是随机生成正态分布的词向量。
            # 也可以是通过自己的语料库训练而得到的词向量。
            if pretrained_embedding is None:
                self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.input_size], -1.0, 1.0),
                                             name="embedding")
            else:
                if embedding_type == 0:
                    self.embedding = tf.constant(pretrained_embedding, name="embedding")
                    self.embedding = tf.cast(self.embedding, tf.float32)
                if embedding_type == 1:
                    self.embedding = tf.Variable(pretrained_embedding, name="embedding", trainable=True)
                    self.embedding = tf.cast(self.embedding, tf.float32)

            self.qid_sequences = tf.nn.embedding_lookup(self.embedding, self.qid_input)
            self.posqid_sequences = tf.nn.embedding_lookup(self.embedding, self.posqid_input)
            self.negqid_sequences = tf.nn.embedding_lookup(self.embedding, self.negqid_input)

        # def conv2d(x, W, stride_h, stride_w, padding='SAME'):
        #     return tf.nn.conv2d(x, W, strides=[1, stride_h, stride_w, 1], padding=padding)
        #
        # def max_pool_2x2(x):
        #     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #
        # def CNN_emb(params,im):
        #     # conv0
        #     conv0 = tf.nn.relu(conv2d(x=im, W=params['conv0W'], stride_h=1, stride_w=1) + params['conv0b'])
        #     pool0 =max_pool_2x2(conv0)
        #
        #     # conv1
        #     conv1 = tf.nn.relu(conv2d(x=pool0, W=params['conv1W'], stride_h=1, stride_w=1) + params['conv1b'])
        #     pool1 = max_pool_2x2(conv1)
        #
        #     # conv2
        #     conv2 = tf.nn.relu(conv2d(x=pool1, W=params['conv2W'], stride_h=1, stride_w=1) + params['conv2b'])
        #     pool2 = max_pool_2x2(conv2)
        #
        #     # conv3
        #     conv3 = tf.nn.relu(conv2d(x=pool2, W=params['conv3W'], stride_h=1, stride_w=1) + params['conv3b'])
        #     pool3 = max_pool_2x2(conv3)
        #
        #     # conv4
        #     conv4 = tf.nn.relu(conv2d(x=pool3, W=params['conv4W'], stride_h=1, stride_w=1) + params['conv4b'])
        #     pool4 = max_pool_2x2(conv4)
        #
        #
        #
        #     shp = pool4.get_shape()
        #     flattened_shape = shp[1].value * shp[2].value * shp[3].value
        #
        #     resh1 = tf.reshape(pool4, [-1, flattened_shape], name="resh1")
        #
        #     im_emb = tf.sigmoid(tf.matmul(resh1, params['conv_emb_W']) + params['conv_emb_b'])
        #
        #     return im_emb
        #
        #
        # cnn_params = np.load(self.datapath + 'CNNmodel/CNNparams.npy').item()
        # with tf.name_scope('CNN'):
        #     params = {}
        #     params['conv0W'] = tf.Variable(cnn_params["conv0"][0], name='conv0W', trainable=True)
        #     params['conv0b'] = tf.Variable(cnn_params["conv0"][1], name='conv0b', trainable=True)
        #     params['conv1W'] = tf.Variable(cnn_params["conv1"][0], name='conv1W', trainable=True)
        #     params['conv1b'] = tf.Variable(cnn_params["conv1"][1], name='conv1b', trainable=True)
        #     params['conv2W'] = tf.Variable(cnn_params["conv2"][0], name='conv2W', trainable=True)
        #     params['conv2b'] = tf.Variable(cnn_params["conv2"][1], name='conv2b', trainable=True)
        #     params['conv3W'] = tf.Variable(cnn_params["conv3"][0], name='conv3W', trainable=True)
        #     params['conv3b'] = tf.Variable(cnn_params["conv3"][1], name='conv3b', trainable=True)
        #     params['conv4W'] = tf.Variable(cnn_params["conv4"][0], name='conv4W', trainable=True)
        #     params['conv4b'] = tf.Variable(cnn_params["conv4"][1], name='conv4b', trainable=True)
        #     params['conv_emb_W'] = tf.Variable(cnn_params["emb"][0], name='conv_emb_W', trainable=True)
        #     params['conv_emb_b'] = tf.Variable(cnn_params["emb"][1], name='conv_emb_b', trainable=True)
        #
        #     qid_im_embed = tf.reshape(CNN_emb(params=params, im=tf.reshape(
        #         tf.reshape(self.qid_images, [-1, self.image_height, self.image_width]),
        #         [-1, self.image_height, self.image_width, 1])),
        #                               [-1, self.max_images_num, self.image_emb_size])
        #     posqid_im_embed = tf.reshape(CNN_emb(params=params, im=tf.reshape(
        #         tf.reshape(self.posqid_images, [-1, self.image_height, self.image_width]),
        #         [-1, self.image_height, self.image_width, 1])),
        #                                  [-1, self.max_images_num, self.image_emb_size])
        #     negqid_im_embed = tf.reshape(CNN_emb(params=params, im=tf.reshape(
        #         tf.reshape(self.negqid_images, [-1, self.image_height, self.image_width]),
        #         [-1, self.image_height, self.image_width, 1])),
        #                                  [-1, self.max_images_num, self.image_emb_size])



        # in_domains = tf.reshape(self.domains,[-1,self.all_domains_num])
        with tf.name_scope('embedding'):
            with tf.device('/cpu:0'):
                embedding = tf.Variable(tf.truncated_normal([self.all_domains_num,self.domain_emb_size], stddev=0.1), name='embedding_domains',dtype=tf.float32)
                qid_domains_embed = tf.nn.embedding_lookup(embedding, self.domains[0,:,:])
                posqid_domains_embed = tf.nn.embedding_lookup(embedding, self.domains[1,:,:])
                negqid_domains_embed = tf.nn.embedding_lookup(embedding, self.domains[2, :, :])

        # embed1 = embed_op(in_domains, name="embed1", n_out=self.domain_emb_size)
        # embed2 = embed_op(embed1, name="embed2", n_out=self.domain_emb_size)
        # domains_embed = tf.reshape(embed2,[3,-1,self.max_domains_num,self.domain_emb_size])
        # qid_domains_embed = domains_embed[0,:,:,:]
        # posqid_domains_embed = domains_embed[1,:, :, :]
        # negqid_domains_embed = domains_embed[2,:, :, :]



        def image_attention(Wa,Va,images_embed,hidden_state,word_embed,mask):
            embed_con = tf.concat([hidden_state,word_embed],axis=1)
            re_embed_con = tf.reshape(tf.reshape(tf.tile(embed_con,[1,self.max_images_num]),[-1,self.max_images_num,self.input_size+self.LSTM_hidden_neurons]),
                                      [-1,self.input_size+self.LSTM_hidden_neurons])
            re_images_embed = tf.reshape(images_embed,[-1,self.image_emb_size])

            in_attn = tf.concat([re_embed_con,re_images_embed],axis=1)
            u = tf.tanh(tf.matmul(in_attn,Wa))
            v = tf.matmul(u,Va)
            exps_temp = tf.reshape(tf.exp(v), [-1, self.max_images_num])
            exps_min = tf.reshape(tf.reduce_min(exps_temp, axis=1), [-1, 1])
            exps = tf.multiply(exps_temp,mask)
            exps_sum = tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
            exps_div = tf.reshape(tf.reduce_max(tf.concat([exps_sum,exps_min],axis=1),axis=1), [-1, 1])
            alphas = exps / exps_div
            res = tf.reduce_sum(images_embed * tf.reshape(alphas, [-1, self.max_images_num, 1]), 1)
            return res

        def domain_attention(Wa,Va,d_embed,hidden_state,word_embed,mask):
            embed_con = tf.concat([hidden_state,word_embed],axis=1)
            re_embed_con = tf.reshape(tf.reshape(tf.tile(embed_con,[1,self.max_domains_num]),[-1,self.max_domains_num,self.input_size+self.LSTM_hidden_neurons]),
                                      [-1,self.input_size+self.LSTM_hidden_neurons])
            re_d_embed = tf.reshape(d_embed,[-1,self.domain_emb_size])

            in_attn = tf.concat([re_embed_con,re_d_embed],axis=1)
            u = tf.tanh(tf.matmul(in_attn,Wa))
            v = tf.matmul(u,Va)
            exps = tf.multiply(tf.reshape(tf.exp(v), [-1, self.max_domains_num]),mask)
            alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
            res = tf.reduce_sum(d_embed * tf.reshape(alphas, [-1, self.max_domains_num, 1]), 1)
            return res

        def make_cos_attention_Vec(states, last_emb,mask):
            # states = [batch, max_text_len, LSTM_hidden_neurons]
            # last_emb = [batch, LSTM_hidden_neurons]
            # return [batch, LSTM_hidden_neurons]
            states_norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(states), axis=2)),[-1,self.max_text_len])
            last_emb_norn = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(last_emb), axis=1)),[-1,1])
            norn_mul = states_norm*last_emb_norn

            last_m = tf.transpose(tf.reshape(last_emb,[-1,1,self.LSTM_hidden_neurons]),[0,2,1])
            cos_res = tf.multiply(tf.div(tf.reshape(tf.matmul(states,last_m),[-1,self.max_text_len]),norn_mul),mask)
            # alpha = tf.div(cos_res,tf.reshape(tf.reduce_sum(cos_res, 1), [-1, 1]))

            Attn_vec = tf.reduce_sum(states * tf.reshape(cos_res, [-1, self.max_text_len, 1]), 1)

            return Attn_vec

        def make_cos_attention_mat(states1, states2,mask1,mask2):
            # states1, states2 = [batch, max_text_len, LSTM_hidden_neurons]
            # return [batch, LSTM_hidden_neurons], [batch, LSTM_hidden_neurons]
            norm1 = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(states1), axis=2)),[-1,self.max_text_len,1])
            norm2 = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(states2), axis=2)),[-1,self.max_text_len,1])


            nor_x1 = tf.multiply(tf.div(states1,norm1),tf.reshape(mask1,[-1,self.max_text_len,1]))
            nor_x2 = tf.multiply(tf.div(states2,norm2),tf.reshape(mask2,[-1,self.max_text_len,1]))

            cos_mat = tf.matmul(nor_x1,tf.transpose(nor_x2,[0,2,1]))
            sum1 = tf.reshape(tf.reduce_sum(cos_mat,axis=2),[-1,self.max_text_len])
            sum2 = tf.reshape(tf.reduce_sum(tf.transpose(cos_mat,[0,2,1]),axis=2),[-1,self.max_text_len])

            # alpha1 = tf.div(sum1,tf.reshape(tf.reduce_sum(sum1, 1), [-1, 1]))
            # alpha2 = tf.div(sum2, tf.reshape(tf.reduce_sum(sum2, 1), [-1, 1]))
            # Attn_vec1 = tf.reduce_sum(states1 * tf.reshape(alpha1, [-1, self.max_text_len, 1]), 1)
            # Attn_vec2 = tf.reduce_sum(states2 * tf.reshape(alpha2, [-1, self.max_text_len, 1]), 1)
            # return Attn_vec1,Attn_vec2
            return sum1, sum2

        with tf.variable_scope('LSTM_ATT_for') as scope:
            Wa_im = tf.Variable(tf.truncated_normal([self.image_emb_size + self.input_size + self.LSTM_hidden_neurons,
                                                     self.soft_attention_size], stddev=0.1))
            Va_im = tf.Variable(tf.truncated_normal([self.soft_attention_size,1], stddev=0.1))
            # ba_im = tf.Variable(tf.constant(0.1,shape = [self.max_images_num]))


            Wa_d = tf.Variable(tf.truncated_normal([self.domain_emb_size + self.input_size + self.LSTM_hidden_neurons,
                                                    self.soft_attention_size], stddev=0.1))
            Va_d = tf.Variable(tf.truncated_normal([self.soft_attention_size,1], stddev=0.1))
            # ba_d = tf.Variable(tf.constant(0.1,shape = [self.max_images_num]))

            self.LSTM = LSTMnet(self.LSTM_hidden_neurons, self.keep_prob)
            initial_state = self.LSTM.cell.zero_state(batch_size, tf.float32)
            print(initial_state)

            # qid_Attn_emb_image =  image_attention(Wa_im,Va_im,qid_im_embed,initial_state[1],self.qid_sequences[:,0,:])
            # qid_Attn_emb_domain = domain_attention(Wa_d, Va_d, qid_domains_embed, initial_state[1], self.qid_sequences[:, 0, :])
            # qid_LSTM_in = tf.concat([self.qid_sequences[:, 0, :],qid_Attn_emb_image,qid_Attn_emb_domain],axis=1)
            # qid_cell_output,qid_state = self.LSTM.cell(qid_LSTM_in,initial_state)

            # qid_hidden_state =[]
            # qid_hidden_state.append(qid_state[1])
            qid_state = initial_state
            posqid_state=initial_state
            negqid_state = initial_state
            qid_hidden_state = []
            posqid_hidden_state = []
            negqid_hidden_state = []
            # scope.reuse_variables()
            for time_step in range(self.max_text_len):
                if time_step> 0:
                    tf.get_variable_scope().reuse_variables()
                # qid
                qid_Attn_emb_image = image_attention(Wa_im, Va_im, qid_im_embed, qid_state[1],
                                                     self.qid_sequences[:, time_step, :], self.qid_images_mask)
                # qid_Attn_emb_image = qid_im_embed[:,0,:]
                qid_Attn_emb_domain = domain_attention(Wa_d, Va_d, qid_domains_embed, qid_state[1],
                                                       self.qid_sequences[:, time_step, :],self.qid_domains_mask)
                qid_LSTM_in = tf.concat(
                    [self.qid_sequences[:, time_step, :], qid_Attn_emb_domain, qid_Attn_emb_image], axis=1)
                qid_cell_output, qid_state = self.LSTM.cell(qid_LSTM_in, qid_state)
                # qid_hidden_state_tmp = qid_cell_output * tf.reshape(self.qid_mask[:,time_step],[-1,1])
                qid_hidden_state.append(qid_cell_output)
                # print(qid_state)

                scope.reuse_variables()
                #pos
                posqid_Attn_emb_image = image_attention(Wa_im, Va_im, posqid_im_embed, posqid_state[1],
                                                        self.posqid_sequences[:, time_step, :], self.posqid_images_mask)
                # posqid_Attn_emb_image = posqid_im_embed[:,0,:]
                posqid_Attn_emb_domain = domain_attention(Wa_d, Va_d, posqid_domains_embed, posqid_state[1],
                                                       self.posqid_sequences[:, time_step, :],self.posqid_domains_mask)
                posqid_LSTM_in = tf.concat([self.posqid_sequences[:, time_step, :], posqid_Attn_emb_domain,posqid_Attn_emb_image],axis=1)
                posqid_cell_output, posqid_state = self.LSTM.cell(posqid_LSTM_in, posqid_state)
                # posqid_hidden_state_tmp =posqid_cell_output*tf.reshape(self.posqid_mask[:,time_step],[-1,1])
                posqid_hidden_state.append(posqid_cell_output)


                #neg
                negqid_Attn_emb_image = image_attention(Wa_im, Va_im, negqid_im_embed, negqid_state[1],
                                                        self.negqid_sequences[:, time_step, :], self.negqid_images_mask)
                # negqid_Attn_emb_image = negqid_im_embed[:,0,:]
                negqid_Attn_emb_domain = domain_attention(Wa_d, Va_d, negqid_domains_embed, negqid_state[1],
                                                          self.negqid_sequences[:, time_step, :],self.negqid_domains_mask)
                negqid_LSTM_in = tf.concat(
                    [self.negqid_sequences[:, time_step, :],  negqid_Attn_emb_domain,negqid_Attn_emb_image], axis=1)
                negqid_cell_output, negqid_state = self.LSTM.cell(negqid_LSTM_in, negqid_state)
                # negqid_hidden_state_tmp = negqid_cell_output * tf.reshape(self.negqid_mask[:,time_step],[-1,1])
                negqid_hidden_state.append(negqid_cell_output)

        qid_hidden_state_merge = tf.reshape(tf.concat(qid_hidden_state,axis=1),[-1,self.max_text_len,self.LSTM_hidden_neurons])
        posqid_hidden_state_merge = tf.reshape(tf.concat(posqid_hidden_state,axis=1),[-1,self.max_text_len,self.LSTM_hidden_neurons])
        negqid_hidden_state_merge = tf.reshape(tf.concat(negqid_hidden_state,axis=1),[-1,self.max_text_len,self.LSTM_hidden_neurons])


        qid_last_state = tf.reduce_sum(
            qid_hidden_state_merge * tf.reshape(self.qid_mask_last, [-1, self.max_text_len, 1]), axis=1)
        posqid_last_state = tf.reduce_sum(
            posqid_hidden_state_merge * tf.reshape(self.posqid_mask_last, [-1, self.max_text_len, 1]), axis=1)
        negqid_last_state = tf.reduce_sum(
            negqid_hidden_state_merge * tf.reshape(self.negqid_mask_last, [-1, self.max_text_len, 1]), axis=1)

        def predict_layer(w1, b1, outw,outb, vec):
            h_fc1 = tf.nn.relu(tf.matmul(vec, w1) + b1)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
            res = tf.matmul(h_fc1_drop, outw) + outb

            # logits = tf.sigmoid(logits)

            return res

        def predict_layer_prob(w1, b1, outw, outb, vec):
            h_fc1 = tf.nn.relu(tf.matmul(vec, w1) + b1)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
            logits = tf.matmul(h_fc1_drop, outw) + outb
            res = tf.sigmoid(logits)

            return res

        attn_pos_vec1 = make_cos_attention_Vec(qid_hidden_state_merge, posqid_last_state, self.qid_mask)
        attn_pos_vec2 = make_cos_attention_Vec(posqid_hidden_state_merge, qid_last_state, self.posqid_mask)
        attn_neg_vec1 = make_cos_attention_Vec(qid_hidden_state_merge, negqid_last_state, self.qid_mask)
        attn_neg_vec2 = make_cos_attention_Vec(negqid_hidden_state_merge, qid_last_state, self.negqid_mask)

        attn_vec1, attn_vec2 = make_cos_attention_mat(qid_hidden_state_merge, posqid_hidden_state_merge,
                                                      self.qid_mask, self.posqid_mask)
        attn_vec3, attn_vec4 = make_cos_attention_mat(qid_hidden_state_merge, negqid_hidden_state_merge,
                                                      self.qid_mask, self.negqid_mask)


        # pairvec_pos = tf.concat([qid_last_state, posqid_last_state,
        #                          tf.reshape(qid_posqid_cos_mat, [-1, self.max_text_len * self.max_text_len])], axis=1)
        # pairvec_neg = tf.concat([qid_last_state, negqid_last_state,
        #                          tf.reshape(qid_negqid_cos_mat, [-1, self.max_text_len * self.max_text_len])], axis=1)
        # pairvec_pos = tf.concat([qid_last_state, posqid_last_state], axis=1)
        # pairvec_neg = tf.concat([qid_last_state, negqid_last_state], axis=1)
        pairvec_pos = tf.concat([qid_last_state, posqid_last_state, attn_pos_vec1, attn_pos_vec2,
                                      attn_vec1, attn_vec2],
                                     axis=1)
        pairvec_neg = tf.concat([qid_last_state, negqid_last_state, attn_neg_vec1, attn_neg_vec2,
                                      attn_vec3, attn_vec4],
                                     axis=1)
        print(pairvec_pos)
        with tf.name_scope('score_layer'):
            #score layer

            # fc_w1 = tf.Variable(tf.truncated_normal([self.LSTM_hidden_neurons*2+self.max_text_len * self.max_text_len, self.score_layer_size1], stddev=0.1), "fc_w1")
            fc_w1 = tf.Variable(tf.truncated_normal(
                [self.LSTM_hidden_neurons * 4+ self.max_text_len*2 , self.score_layer_size1],
                stddev=0.1), "fc_w1")
            fc_b1 = tf.Variable(tf.constant(0.1, shape=[self.score_layer_size1]), "fc_b1")

            output_w = tf.Variable(tf.truncated_normal([self.score_layer_size1, 1], stddev=0.1), "out_w")
            output_b = tf.Variable(tf.constant(0.1, shape=[1]), "out_b")

            self.pred_pos = predict_layer_prob(fc_w1, fc_b1,output_w, output_b, pairvec_pos)
            self.pred_neg = predict_layer_prob(fc_w1, fc_b1,output_w, output_b, pairvec_neg)

        hold_score = self.margin - self.pred_pos + self.pred_neg

        batch_zero = tf.zeros([batch_size, 1])

        max_score = tf.concat([hold_score, batch_zero], axis=1)

        self.loss = tf.reduce_sum(tf.reduce_max(max_score, axis=1))

        rl2 = self.l2_reg * sum(
            tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            # if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )
        self.loss += rl2

        self.lr = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), 2)

        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.apply_gradients(list(zip(self.grads, trainable_vars)))

        print("=" * 50)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
        print("=" * 50)



    def step(self, sess, qid_materials, posqid_materials, negqid_materials, domains_mat, is_train):

        input_feed = {self.qid_input: qid_materials['sentences'],
                      self.qid_mask: qid_materials['mask'],
                      self.qid_mask_last: qid_materials['last_mask'],
                      self.posqid_input: posqid_materials['sentences'],
                      self.posqid_mask: posqid_materials['mask'],
                      self.posqid_mask_last: posqid_materials['last_mask'],
                      self.negqid_input: negqid_materials['sentences'],
                      self.negqid_mask: negqid_materials['mask'],
                      self.negqid_mask_last: negqid_materials['last_mask'],
                      self.qid_images: qid_materials['images'],
                      self.posqid_images: posqid_materials['images'],
                      self.negqid_images: negqid_materials['images'],
                      # self.images :images_mat,
                      self.domains:domains_mat,
                      self.qid_images_mask:qid_materials['images_mask'],
                      self.qid_domains_mask: qid_materials['domains_mask'],
                      self.posqid_images_mask: posqid_materials['images_mask'],
                      self.posqid_domains_mask: posqid_materials['domains_mask'],
                      self.negqid_images_mask: negqid_materials['images_mask'],
                      self.negqid_domains_mask: negqid_materials['domains_mask'],
                    }

        if is_train:
            input_feed[self.keep_prob] = self.keep_prob_value
            train_loss, _, pred_pos,pred_neg= sess.run(
                [self.loss, self.train_op, self.pred_pos,self.pred_neg], input_feed)
            print(pred_pos[0])
            print(pred_neg[0])
            # print(qid_materials['images_mask'][0])
            # print(posqid_materials['images_mask'][0])
            # print(negqid_materials['images_mask'][0])
            return train_loss
        else:
            input_feed[self.keep_prob] = 1
            pred_all = sess.run(self.pred_pos, input_feed)
            return pred_all

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))




def load_word2vec_matrix(w2vpath, embedding_size):
    """
    Return the word2vec model matrix.
    :param vocab_size: The vocab size of the word2vec model file
    :param embedding_size: The embedding size
    :return: The word2vec model matrix
    """
    word2vec_file = w2vpath
    wvmodel ={}
    if os.path.isfile(word2vec_file):
        model = gensim.models.Word2Vec.load(word2vec_file)
        vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
        vocab_size = len(vocab)
        print(vocab_size)
        vector = np.zeros([vocab_size+1, embedding_size])
        for key, value in vocab.items():
            wvmodel[key] = value+1
            if len(key) > 0:
                vector[value+1] = model[key]
        vector[0] = np.random.uniform(-5.0, 5.0, embedding_size)
        return vector,wvmodel,vocab_size
    else:
        logging.info("✘ The word2vec file doesn't exist. "
                     "Please use function <create_vocab_size(embedding_size)> to create it!")

def load_images(imagepath,image_emb_path):
    images= {}
    images_emb_map ={}
    fin = open(imagepath)
    totalnum = 0
    for eachLine in fin:
        totalnum = totalnum +1
        print(totalnum)
        pid = eachLine.replace('\n', '').split('\t')
        qid = pid[0]
        piclist = []
        pics = pid[1].split(' ')
        for pic in pics:
            picpath = image_emb_path+ qid +'/'+pic+'.npy'
            if os.path.exists(picpath):
                piclist.append(pic)
                images_emb_map[qid +'/'+pic] = np.load(picpath)
        images[qid] = piclist
    fin.close()
    return images,images_emb_map

def Rank_precision_recall_at(score_label_sorted,all_posnum,topk):
    pr_score =0.0
    recall_score = 0.0
    qidnum = 0
    F1_score = 0.0
    qid_result = {}
    for qid,score_labal in score_label_sorted.items():
        qidnum += 1
        len1 = min(len(score_labal),topk)
        hit = 0
        for i in range(len1):
            if score_labal[i][1] == 1:
                hit +=1
        qid_pr = float(hit)/float(topk)
        pr_score += qid_pr
        qid_recall = float(hit)/float(all_posnum[qid])
        recall_score += qid_recall
        qid_F1 = 2 * float(hit) / (float(topk) + float(all_posnum[qid]))
        F1_score += qid_F1
        qid_result[qid] = [qid_pr,qid_recall,qid_F1]


    res_pr = pr_score / qidnum
    res_rc = recall_score / qidnum

    F1_value = F1_score / qidnum

    return res_pr, res_rc, F1_value,qid_result



def run(sess,flags,filepath,output,modelpath):
    epochnum = 50
    print(flags.train)
    embedding_size = 100
    image_emb_size = 100
    domain_emb_size = 100
    print('***********load_word2vec***********')
    word2vec_matrix, wvmodel, vocab_size = load_word2vec_matrix(
        filepath + 'word2vec_' + str(embedding_size) + '/w2vmodel',
        embedding_size)
    # config and create model
    config = {"max_images_num": 10,
              "max_domains_num": 10,
              "max_text_len": 300,
              "datapath": filepath,
              "all_domains_num": 1047,
              "batch_size": 128,
              "vocab_size": vocab_size,
              "input_size": embedding_size,
              "image_emb_size": image_emb_size,
              "domain_emb_size": domain_emb_size,
              "keep_prob": 0.8,
              "margin": 0.8,
              'l2_reg': 0.00004,
              'score_layer_size1': 200
              }
    print('***********initial model***********')
    model = MANN(config=config,pretrained_embedding =word2vec_matrix)
    # sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=50)
    saver.restore(sess, flags.ModelPath)
    # sess.run(tf.global_variables_initializer())
    print('***********images_emb_map***********')
    images, images_emb_map = load_images(filepath + 'images_list.txt',
                                         filepath + 'image_emb_' + str(image_emb_size) + '/')

    if flags.train:
        outfile = open(output + 'result-MANN-08', 'w')
        print('***********load traindata**********')
        traindata = Data(config=config,images=images,images_emb_map = images_emb_map,is_train=True)

        traindata.load_domains(filepath+'knowledge_num_list.txt')
        print('***********load testdata***********')
        testdata = Data(config=config,images=images,images_emb_map = images_emb_map,is_train=False)
        testdata.load_domains(filepath+'knowledge_num_list.txt')

        lr = 0.2
        lr_decay = 0.9
        print('***********begin train***********')
        # run epoch
        for epoch in range(epochnum):
            # train
            model.assign_lr(sess, lr * lr_decay ** epoch)
            overall_loss = 0
            st = time.time()
            traindata.reload_train_data_with_num(trainpath= filepath+'Train.json',compare_neg_num=2)
            traindata.shuffle()
            print(traindata.batch_num)
            while not traindata.end:
                _,qid_materials, posqid_materials, negqid_materials, domains_mat, data_label = traindata.next_batch(wvmodel=wvmodel)
                # model.set_batchsize(batchnum)
                loss = model.step(sess, qid_materials, posqid_materials, negqid_materials,
                                  domains_mat,
                                  is_train=True)
                overall_loss += loss
                print(("\r loss:{0}, time spent:{1}s".format(loss, time.time() - st)))
                sys.stdout.flush()

            print(("\r overall_loss:{0}, time spent:{1}s".format(overall_loss, time.time() - st)))
            sys.stdout.flush()
            if epoch >= 0:
                saver.save(sess, modelpath+"MANN08/MANN"+str(epoch))
                # test
                testdata.reset()
                pred_label =[]
                while not testdata.end:
                    batch_data,qid_materials, posqid_materials, negqid_materials, domains_mat, data_label = testdata.next_batch(wvmodel=wvmodel)
                    pred_all = model.step(sess,qid_materials, posqid_materials, negqid_materials,
                                      domains_mat,
                                      is_train=False)
                    batch1 = batch_data.tolist()
                    batchnum = len(batch1)
                    for idx in range(batchnum):
                        pred_label.append((batch1[idx][0],batch1[idx][1],batch1[idx][2],pred_all[idx],data_label[idx]))

                # compute metrics
                qid_score_label_tmp = {}
                qid_all_posnum = {}
                for eachpred in pred_label:
                    if eachpred[0] not in qid_score_label_tmp:
                        qid_score_label_tmp[eachpred[0]] = []

                    qid_score_label_tmp[eachpred[0]].append((eachpred[3],eachpred[4]))

                    if eachpred[0] not in qid_all_posnum:
                        qid_all_posnum[eachpred[0]] = 0
                    if eachpred[4] == 1:
                        qid_all_posnum[eachpred[0]] += 1

                qid_score_label = {}
                for qid, score_labal in qid_score_label_tmp.items():
                    qid_score_label[qid] = sorted(score_labal,key=lambda asd:asd[0],reverse = True)

                topn = 1
                while topn <= 10:
                    precision_at, recall_at, F1_value_at,_ = Rank_precision_recall_at(qid_score_label, qid_all_posnum,topn)
                    print(("\n epoch = {0}, top n ={1},precision_at1={2}, recall_at1={3}, F1_value_at1={4}".format(
                        epoch,
                        topn,
                        precision_at,
                        recall_at,
                        F1_value_at)))
                    outfile.write(
                        ("\n epoch = {0},top n ={1}, precision_at, recall_at, F1_value_at\n{2}\t{3}\t{4} ".format(
                            epoch, topn,
                            precision_at,
                            recall_at,
                            F1_value_at)))
                    topn = topn + 1

                # precision_at1, recall_at1, F1_value_at1,qid_result_at1 = Rank_precision_recall_at(qid_score_label, qid_all_posnum, 1)
                # precision_at2, recall_at2, F1_value_at2,qid_result_at2 = Rank_precision_recall_at(qid_score_label, qid_all_posnum, 2)
                # precision_at3, recall_at3, F1_value_at3,qid_result_at3 = Rank_precision_recall_at(qid_score_label, qid_all_posnum, 3)
                # precision_at5, recall_at5, F1_value_at5,qid_result_at5 = Rank_precision_recall_at(qid_score_label, qid_all_posnum, 5)
                # precision_at10, recall_at10, F1_value_at10,qid_result_at10 = Rank_precision_recall_at(qid_score_label, qid_all_posnum, 10)
                # print(("\n epoch = {0}, precision_at1={1}, recall_at1={2}, F1_value_at1={3}".format(epoch, precision_at1,
                #                                                                                     recall_at1, F1_value_at1)))
                # outfile.write(("\n epoch = {0}, precision_at1, recall_at1, F1_value_at1\n{1}\t{2}\t{3} ".format(epoch,
                #                                                                                                 precision_at1,
                #                                                                                                 recall_at1,
                #                                                                                                 F1_value_at1)))
                # print(("\n epoch = {0}, precision_at2={1}, recall_at2={2}, F1_value_at2={3}".format(epoch, precision_at2,
                #                                                                                     recall_at2, F1_value_at2)))
                # outfile.write(("\n epoch = {0}, precision_at2, recall_at2, F1_value_at2\n{1}\t{2}\t{3}".format(epoch,
                #                                                                                                precision_at2,
                #                                                                                                recall_at2,
                #                                                                                                F1_value_at2)))
                # print(("\n epoch = {0}, precision_at3={1}, recall_at3={2}, F1_value_at3={3}".format(epoch, precision_at3,
                #                                                                                     recall_at3, F1_value_at3)))
                # outfile.write(("\n epoch = {0}, precision_at3, recall_at3, F1_value_at3\n{1}\t{2}\t{3}".format(epoch,
                #                                                                                                precision_at3,
                #                                                                                                recall_at3,
                #                                                                                                F1_value_at3)))
                # print(("\n epoch = {0}, precision_at5={1}, recall_at5={2}, F1_value_at5={3}".format(epoch, precision_at5,
                #                                                                                     recall_at5, F1_value_at5)))
                # outfile.write(("\n epoch = {0}, precision_at5, recall_at5, F1_value_at5\n{1}\t{2}\t{3}".format(epoch,
                #                                                                                                precision_at5,
                #                                                                                                recall_at5,
                #                                                                                                F1_value_at5)))
                # print(("\n epoch = {0}, precision_at10={1}, recall_at10={2}, F1_value_at10={3}".format(epoch, precision_at10,
                #                                                                                        recall_at10,
                #                                                                                        F1_value_at10)))
                # outfile.write(("\n epoch = {0}, precision_at10, recall_at10, F1_value_at10\n{1}\t{2}\t{3}".format(epoch,
                #                                                                                                   precision_at10,
                #                                                                                                   recall_at10,
                #                                                                                                   F1_value_at10)))
        outfile.close()
        # predfile.close()
    else:
        outfile = open(output + 'test-result-MANN', 'w')
        # predfile = open(output + 'all-test-result-MANN.json', 'w')
        saver.restore(sess, flags.ModelPath)
        testdata = Data(config=config, images=images, images_emb_map=images_emb_map, is_train=False)
        testdata.load_domains(filepath + 'knowledge_num_list.txt')
        testdata.reset()
        epoch = 0
        pred_label = []
        while not testdata.end:
            batch_data, qid_materials, posqid_materials, negqid_materials, domains_mat, data_label = testdata.next_batch(wvmodel=wvmodel)
            pred_all = model.step(sess, qid_materials, posqid_materials, negqid_materials,
                                  domains_mat,
                                  is_train=False)
            batch1 = batch_data.tolist()
            batchnum = len(batch1)
            for idx in range(batchnum):
                pred_label.append((batch1[idx][0], batch1[idx][1], batch1[idx][2], pred_all[idx], data_label[idx]))

        # compute metrics
        qid_score_label_tmp = {}
        qid_all_posnum = {}
        for eachpred in pred_label:
            if eachpred[0] not in qid_score_label_tmp:
                qid_score_label_tmp[eachpred[0]] = []

            qid_score_label_tmp[eachpred[0]].append((eachpred[3], eachpred[4]))

            if eachpred[0] not in qid_all_posnum:
                qid_all_posnum[eachpred[0]] = 0
            if eachpred[4] == 1:
                qid_all_posnum[eachpred[0]] += 1

        qid_score_label = {}
        for qid, score_labal in qid_score_label_tmp.items():
            qid_score_label[qid] = sorted(score_labal, key=lambda asd: asd[0], reverse=True)

        topn = 1
        while topn <= 10:
            precision_at, recall_at, F1_value_at,_ = Rank_precision_recall_at(qid_score_label, qid_all_posnum,topn)
            print(("\n epoch = {0}, top n ={1},precision_at1={2}, recall_at1={3}, F1_value_at1={4}".format(
                epoch,
                topn,
                precision_at,
                recall_at,
                F1_value_at)))
            outfile.write(
                ("\n epoch = {0},top n ={1}, precision_at, recall_at, F1_value_at\n{2}\t{3}\t{4} ".format(
                    epoch, topn,
                    precision_at,
                    recall_at,
                    F1_value_at)))
            topn = topn + 1

        # qid_json ={}
        # for qid,res in qid_result_at1.items():
        #     qid_json['qid'] = qid
        #     qid_json['at1'] = res
        #     qid_json['at2'] = qid_result_at2[qid]
        #     qid_json['at3'] = qid_result_at3[qid]
        #     qid_json['at4'] = qid_result_at10[qid]
        #     qid_json['at5'] = qid_result_at5[qid]
        #     json.dump(qid_json, predfile, ensure_ascii=False)
        #     predfile.write('\n')

        outfile.close()
        # predfile.close()



def main(flags):
    #
    # TODO: Any code here.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    inputstr = "/home/huangzai/MANN/dataset1/data/"
    outputstr = "/home/huangzai/MANN/dataset1/result/"
    modelstr = "/home/huangzai/MANN/dataset1/model/"
    with tf.Session(config=config) as session:
        # session.run(tf.global_variables_initializer())
        #
        # TODO: Any code here.
        run(sess=session,flags=flags,filepath = inputstr,output = outputstr,modelpath = modelstr)
    return 0

## python3 QPS_NoAttn.py -gpu 0
if __name__ == '__main__':
    global_flags = gflags.FLAGS
    gflags.DEFINE_boolean('help', False, 'Show this help.')
    gflags.DEFINE_string('gpu', '3', 'Which GPU to use.')
    gflags.DEFINE_boolean('train', True, 'train type.')
    gflags.DEFINE_boolean('case', False, 'case study.')
    gflags.DEFINE_string('ModelPath', '/home/huangzai/MANN/dataset1/model/MANN-0119/MANN11', 'model path.')
    #
    # TODO: Other FLAGS here.
    global_flags(sys.argv)
    if global_flags.help:
        print(global_flags.main_module_help())
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = global_flags.gpu
    exit(main(global_flags))


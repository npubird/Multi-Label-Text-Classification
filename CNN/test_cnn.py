# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import numpy as np
import tensorflow as tf
from utils import data_helpers as dh

# Parameters
# ==================================================

logger = dh.logger_fn('tflog', 'logs/test-{0}.log'.format(time.asctime()))

MODEL = input("☛ Please input the model file you want to test, it should be like(1490175368): ")

while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input('✘ The format of your input is illegal, it should be like(1490175368), please re-input: ')
logger.info('✔︎ The format of your input is legal, now loading to next step...')

CLASS_BIND = input("☛ Use Class Bind or Not?(Y/N) \n")
while not (CLASS_BIND.isalpha() and CLASS_BIND.upper() in ['Y', 'N']):
    CLASS_BIND = input('✘ The format of your input is illegal, please re-input: ')
logger.info('✔︎ The format of your input is legal, now loading to next step...')

CLASS_BIND = CLASS_BIND.upper()

TRAININGSET_DIR = '../data/Train.json'
VALIDATIONSET_DIR = '../data/Validation_bind.json'
TESTSET_DIR = '../data/Test.json'
MODEL_DIR = 'runs/' + MODEL + '/checkpoints/'
SAVE_DIR = 'results/' + MODEL

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data")
tf.flags.DEFINE_string("test_data_file", TESTSET_DIR, "Data source for the test data")
tf.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")
tf.flags.DEFINE_string("use_classbind_or_not", CLASS_BIND, "Use the class bind info or not.")

# Model Hyperparameters
tf.flags.DEFINE_integer("pad_seq_len", 150, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_classes", 367, "Number of labels (depends on the task)")
tf.flags.DEFINE_integer("top_num", 3, "Number of top K prediction classes (default: 3)")

# Test Parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def test_cnn():
    """Test CNN model."""

    # Load data
    logger.info("✔ Loading data...")
    logger.info('Recommended padding Sequence length is: {0}'.format(FLAGS.pad_seq_len))

    logger.info('✔︎ Test data processing...')
    test_data = dh.load_data_and_labels(FLAGS.test_data_file, FLAGS.num_classes, FLAGS.embedding_dim)

    logger.info('✔︎ Test data padding...')
    x_test, y_test = dh.pad_data(test_data, FLAGS.pad_seq_len)
    y_test_bind = test_data.labels_bind

    # Build vocabulary
    VOCAB_SIZE = dh.load_vocab_size(FLAGS.embedding_dim)
    pretrained_word2vec_matrix = dh.load_word2vec_matrix(VOCAB_SIZE, FLAGS.embedding_dim)

    # Load cnn model
    logger.info("✔ Loading model...")
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # pre-trained word2vec
            pretrained_embedding = graph.get_operation_by_name("embedding/embedding").outputs[0]

            # Tensors we want to evaluate
            logits = graph.get_operation_by_name("output/logits").outputs[0]
            topKPreds = graph.get_operation_by_name("output/topKPreds").outputs[0]
            loss = graph.get_operation_by_name("loss/loss").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = 'output/logits|output/scores|output/topKPreds'

            # Save the .pb model file
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, 'graph', 'graph-cnn-{0}.pb'.format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(zip(x_test, y_test, y_test_bind)), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predicitons = np.empty(shape=(0, FLAGS.top_num))
            all_topKPreds = np.empty(shape=(0, FLAGS.top_num))

            eval_loss, eval_rec, eval_acc, eval_counter = 0.0, 0.0, 0.0, 0

            for batch_test in batches:
                x_batch_test, y_batch_test, y_batch_test_bind = zip(*batch_test)
                feed_dict = {
                    input_x: x_batch_test,
                    input_y: y_batch_test,
                    dropout_keep_prob: 1.0,
                    is_training: False
                }
                batch_logits = sess.run(logits, feed_dict)

                batch_topKPreds = sess.run(topKPreds, feed_dict)
                all_topKPreds = np.vstack((all_topKPreds, batch_topKPreds))

                batch_loss = sess.run(loss, feed_dict)

                if FLAGS.use_classbind_or_not == 'Y':
                    predicted_labels = dh.get_label_using_logits_and_classbind(
                        batch_logits, y_batch_test_bind, top_number=FLAGS.top_num)
                if FLAGS.use_classbind_or_not == 'N':
                    predicted_labels = dh.get_label_using_logits(batch_logits, top_number=FLAGS.top_num)

                all_predicitons = np.vstack((all_predicitons, predicted_labels))

                cur_rec, cur_acc = 0.0, 0.0
                for index, predicted_label in enumerate(predicted_labels):
                    rec_inc, acc_inc = dh.cal_rec_and_acc(predicted_label, y_batch_test[index])
                    cur_rec, cur_acc = cur_rec + rec_inc, cur_acc + acc_inc

                cur_rec = cur_rec / len(y_batch_test)
                cur_acc = cur_acc / len(y_batch_test)

                eval_rec, eval_acc, eval_counter = eval_rec + cur_rec, eval_acc + cur_acc, eval_counter + 1
                logger.info("✔︎ Test batch {0}: loss {1:g}, recall {2:g}, accuracy {2:g}."
                            .format(eval_counter, batch_loss, cur_rec, cur_acc))

            eval_rec = float(eval_rec / eval_counter)
            eval_acc = float(eval_acc / eval_counter)
            logger.info("☛ All Test Dataset: Recall {0:g}, Accuracy {1:g}".format(eval_rec, eval_acc))
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            dh.create_preditction_file(file=SAVE_DIR + '/predictions.json', data_id=test_data.testid,
                                       all_predict_labels=all_predicitons, all_predict_values=all_topKPreds)

    logger.info("✔ Done.")


if __name__ == '__main__':
    test_cnn()

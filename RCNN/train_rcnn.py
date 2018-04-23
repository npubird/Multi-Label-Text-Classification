# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import datetime
import logging
import numpy as np
import tensorflow as tf

from utils import data_helpers as dh
from text_rcnn import TextRCNN
from tensorboard.plugins import projector

# Parameters
# ==================================================

TRAIN_OR_RESTORE = input("☛ Train or Restore?(T/R) \n")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input('✘ The format of your input is illegal, please re-input: ')
logging.info('✔︎ The format of your input is legal, now loading to next step...')

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

CLASS_BIND = input("☛ Use Class Bind or Not?(Y/N) \n")
while not (CLASS_BIND.isalpha() and CLASS_BIND.upper() in ['Y', 'N']):
    CLASS_BIND = input('✘ The format of your input is illegal, please re-input: ')
logging.info('✔︎ The format of your input is legal, now loading to next step...')

CLASS_BIND = CLASS_BIND.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn('tflog', 'logs/training-{0}.log'.format(time.asctime()))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn('tflog', 'logs/restore-{0}.log'.format(time.asctime()))

TRAININGSET_DIR = '../data/Train.json'
VALIDATIONSET_DIR = '../data/Validation_bind.json'
METADATA_DIR = '../data/metadata.tsv'

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data.")
tf.flags.DEFINE_string("metadata_file", METADATA_DIR, "Metadata file for embedding visualization"
                                                      "(Each line is a word segment in metadata_file).")
tf.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")
tf.flags.DEFINE_string("use_classbind_or_not", CLASS_BIND, "Use the class bind info or not.")

# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 0.001, "The learning rate (default: 0.001)")
tf.flags.DEFINE_integer("pad_seq_len", 150, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_classes", 367, "Number of labels (depends on the task)")
tf.flags.DEFINE_integer("top_num", 1, "Number of top K prediction classes (default: 3)")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 5000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 0.5, "Rate of decay for learning rate.")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def train_rcnn():
    """Training RCNN model."""

    # Load sentences, labels, and training parameters
    logger.info('✔︎ Loading data...')

    logger.info('✔︎ Training data processing...')
    train_data = dh.load_data_and_labels(FLAGS.training_data_file, FLAGS.num_classes, FLAGS.embedding_dim)

    logger.info('✔︎ Validation data processing...')
    validation_data = \
        dh.load_data_and_labels(FLAGS.validation_data_file, FLAGS.num_classes, FLAGS.embedding_dim)

    logger.info('Recommended padding Sequence length is: {0}'.format(FLAGS.pad_seq_len))

    logger.info('✔︎ Training data padding...')
    x_train, y_train = dh.pad_data(train_data, FLAGS.pad_seq_len)

    logger.info('✔︎ Validation data padding...')
    x_validation, y_validation = dh.pad_data(validation_data, FLAGS.pad_seq_len)

    y_validation_bind = validation_data.labels_bind

    # Build vocabulary
    VOCAB_SIZE = dh.load_vocab_size(FLAGS.embedding_dim)
    pretrained_word2vec_matrix = dh.load_word2vec_matrix(VOCAB_SIZE, FLAGS.embedding_dim)

    # Build a graph and rcnn object
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rcnn = TextRCNN(
                sequence_length=FLAGS.pad_seq_len,
                num_classes=FLAGS.num_classes,
                batch_size=FLAGS.batch_size,
                vocab_size=VOCAB_SIZE,
                embedding_size=FLAGS.embedding_dim,
                embedding_type=FLAGS.embedding_type,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                pretrained_embedding=pretrained_word2vec_matrix)

            # Define training procedure
            # learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate, global_step=cnn.global_step,
            #                                            decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate,
            #                                            staircase=True)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
                grads_and_vars = optimizer.compute_gradients(rcnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=rcnn.global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            if FLAGS.train_or_restore == 'R':
                MODEL = input("☛ Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input('✘ The format of your input is illegal, please re-input: ')
                logger.info('✔︎ The format of your input is legal, now loading to next step...')

                checkpoint_dir = 'runs/' + MODEL + '/checkpoints/'

                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("✔︎ Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                logger.info("✔︎ Writing to {0}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rcnn.loss)
            # acc_summary = tf.summary.scalar("accuracy", rcnn.accuracy)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            if FLAGS.train_or_restore == 'R':
                # Load rcnn model
                logger.info("✔ Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Embedding visualization config
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = 'embedding'
                embedding_conf.metadata_path = FLAGS.metadata_file

                projector.visualize_embeddings(train_summary_writer, config)
                projector.visualize_embeddings(validation_summary_writer, config)

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, 'embedding', 'embedding.ckpt'))

            current_step = sess.run(rcnn.global_step)

            def batch_iter(data, batch_size, num_epochs, shuffle=True):
                """
                The function <batch_iter> in data_helpers.py will create the data batch
                which has not exactly batch size since that we have to overwrite the function for rcnn
                Because rcnn need the all batches has the exact batch size otherwise will raise error
                """
                data = np.array(data)
                data_size = len(data)
                # Just the diff in var num_batches_per_epoch
                # Do not plus one in there
                # Because we need to drop the last batch in case it has not exactly batch_size
                num_batches_per_epoch = int((data_size - 1) / batch_size)
                for epoch in range(num_epochs):
                    # Shuffle the data at each epoch
                    if shuffle:
                        shuffle_indices = np.random.permutation(np.arange(data_size))
                        shuffled_data = data[shuffle_indices]
                    else:
                        shuffled_data = data
                    for batch_num in range(num_batches_per_epoch):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size)
                        yield shuffled_data[start_index:end_index]

            def train_step(x_batch, y_batch):
                """A single training step"""
                feed_dict = {
                    rcnn.input_x: x_batch,
                    rcnn.input_y: y_batch,
                    rcnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    rcnn.is_training: True
                }
                _, step, summaries, loss = sess.run(
                    [train_op, rcnn.global_step, train_summary_op, rcnn.loss], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logger.info("{0}: step {1}, loss {2:g}".format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_validation, y_validation, y_validation_bind, writer=None):
                """Evaluates model on a validation set"""
                batches_validation = batch_iter(
                    list(zip(x_validation, y_validation, y_validation_bind)), FLAGS.batch_size, FLAGS.num_epochs)
                eval_loss, eval_rec, eval_acc, eval_counter = 0.0, 0.0, 0.0, 0
                for batch_validation in batches_validation:
                    x_batch_validation, y_batch_validation, y_batch_validation_bind = zip(*batch_validation)
                    feed_dict = {
                        rcnn.input_x: x_batch_validation,
                        rcnn.input_y: y_batch_validation,
                        rcnn.dropout_keep_prob: 1.0,
                        rcnn.is_training: False
                    }
                    step, summaries, logits, cur_loss = sess.run(
                        [rcnn.global_step, validation_summary_op, rcnn.logits, rcnn.loss], feed_dict)

                    if FLAGS.use_classbind_or_not == 'Y':
                        predicted_labels = dh.get_label_using_logits_and_classbind(
                            logits, y_batch_validation_bind, top_number=FLAGS.top_num)
                    if FLAGS.use_classbind_or_not == 'N':
                        predicted_labels = dh.get_label_using_logits(logits, top_number=FLAGS.top_num)

                    cur_rec, cur_acc = 0.0, 0.0
                    for index, predicted_label in enumerate(predicted_labels):
                        rec_inc, acc_inc = dh.cal_rec_and_acc(predicted_label, y_batch_validation[index])
                        cur_rec, cur_acc = cur_rec + rec_inc, cur_acc + acc_inc

                    cur_rec = cur_rec / len(y_batch_validation)
                    cur_acc = cur_acc / len(y_batch_validation)

                    eval_loss, eval_rec, eval_acc, eval_counter = eval_loss + cur_loss, eval_rec + cur_rec, \
                                                                  eval_acc + cur_acc, eval_counter + 1
                    logger.info("✔︎ validation batch {0} finished.".format(eval_counter))

                    if writer:
                        writer.add_summary(summaries, step)

                eval_loss = float(eval_loss / eval_counter)
                eval_rec = float(eval_rec / eval_counter)
                eval_acc = float(eval_acc / eval_counter)

                return eval_loss, eval_rec, eval_acc

            # Generate batches
            batches_train = batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            for batch_train in batches_train:
                x_batch_train, y_batch_train = zip(*batch_train)
                train_step(x_batch_train, y_batch_train)
                current_step = tf.train.global_step(sess, rcnn.global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    eval_loss, eval_rec, eval_acc = validation_step(x_validation, y_validation, y_validation_bind,
                                                                    writer=validation_summary_writer)
                    time_str = datetime.datetime.now().isoformat()
                    logger.info("{0}: step {1}, loss {2:g}, rec {3:g}, acc {4:g}"
                                .format(time_str, current_step, eval_loss, eval_rec, eval_acc))

                if current_step % FLAGS.checkpoint_every == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("✔︎ Saved model checkpoint to {0}\n".format(path))

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    train_rcnn()

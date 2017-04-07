from __future__ import print_function
from __future__ import division

import sys
import tensorflow as tf
import numpy as np
import tf_utils

'''
' this code is adapted from Emma Strubell's implementation of the BiLSTM model
'''
class LSTM(object):

    """
    A normal LSTM for text classification.
    """
    def __init__(self, num_classes, vocab_size, shape_domain_size, char_domain_size, char_size,
            embedding_size, shape_size, nonlinearity, viterbi, hidden_dim, char_embeddings, embeddings=None,
            use_geometric_feats=False):

        self.num_classes = num_classes
        self.shape_domain_size = shape_domain_size
        self.char_domain_size = char_domain_size
        self.char_size = char_size
        self.embedding_size = embedding_size
        self.shape_size = shape_size
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        self.char_embeddings = char_embeddings

        # word embedding input
        self.input_x1 = tf.placeholder(tf.int64, [None, None], name="input_x1")

        # shape embedding input
        self.input_x2 = tf.placeholder(tf.int64, [None, None], name="input_x2")

        # geometric inputs
        self.widths = tf.placeholder(tf.float64, [None, None, 1], name="widths")
        self.heights = tf.placeholder(tf.float64, [None, None, 1], name="heights")
        self.wh_ratios = tf.placeholder(tf.float64, [None, None, 1], name="wh_ratios")
        self.x_coords = tf.placeholder(tf.float64, [None, None, 1], name="x_coords")
        self.y_coords = tf.placeholder(tf.float64, [None, None, 1], name="y_coords")
        self.pages = tf.placeholder(tf.int64, [None, None, 1], name="pages")
        self.lines = tf.placeholder(tf.int64, [None, None, 1], name="lines")
        self.zones = tf.placeholder(tf.int64, [None, None, 1], name="zones")

        # labels
        self.input_y = tf.placeholder(tf.int64, [None, None], name="input_y")

        # padding mask
        self.input_mask = tf.placeholder(tf.float32, [None, None], name="input_mask")

        self.batch_size = tf.placeholder(tf.int32, None, name="batch_size")

        self.max_seq_len = tf.placeholder(tf.int32, None, name="max_seq_len")

        # sequence lengths
        self.sequence_lengths = tf.placeholder(tf.int32, [None, None], name="sequence_lengths")

        # dropout and l2 penalties
        self.middle_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="middle_dropout_keep_prob")
        self.hidden_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="hidden_dropout_keep_prob")
        self.input_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="input_dropout_keep_prob")
        self.word_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="word_dropout_keep_prob")

        self.l2_penalty = tf.placeholder_with_default(0.0, [], name="l2_penalty")

        self.projection = tf.placeholder_with_default(False, [], name="projection")

        self.drop_penalty = tf.placeholder_with_default(0.0, [], name="drop_penalty")

        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

        # set the pad token to a constant 0 vector
        self.word_zero_pad = tf.constant(0.0, dtype=tf.float32, shape=[1, embedding_size])
        self.shape_zero_pad = tf.constant(0.0, dtype=tf.float32, shape=[1, shape_size])
        self.char_zero_pad = tf.constant(0.0, dtype=tf.float32, shape=[1, char_size])

        self.use_characters = char_size != 0
        self.use_shape = shape_size != 0
        self.use_geometric_feats = use_geometric_feats

        # Embedding layer
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        word_embeddings_shape = (vocab_size - 1, embedding_size)
        self.w_e = tf_utils.initialize_embeddings(word_embeddings_shape, name="w_e", pretrained=embeddings)

        nonzero_elements = tf.not_equal(self.sequence_lengths, tf.zeros_like(self.sequence_lengths))
        count_nonzero_per_row = tf.reduce_sum(tf.to_int32(nonzero_elements), reduction_indices=1)
        # todo: this is the wrong type or something?
        # self.flat_sequence_lengths = tf.cast(tf.add(tf.reduce_sum(self.sequence_lengths, 1), tf.scalar_mul(2, count_nonzero_per_row)), tf.int64)
        # print(self.flat_sequence_lengths.get_shape())
        self.flat_sequence_lengths = tf.reshape(self.sequence_lengths, [-1])

        # tf.Print(self.flat_sequence_lengths, [self.flat_sequence_lengths.type])

        self.unflat_scores = self.forward(self.input_x1, self.input_x2, self.max_seq_len,
                                          self.hidden_dropout_keep_prob,
                                          self.input_dropout_keep_prob, self.middle_dropout_keep_prob, reuse=False)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            labels = tf.cast(self.input_y, 'int32')
            if viterbi:
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.unflat_scores, labels, self.flat_sequence_lengths)
                self.transition_params = transition_params
                self.loss = tf.reduce_mean(-log_likelihood)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.unflat_scores, labels=labels)
                masked_losses = tf.mul(losses, self.input_mask)
                self.loss = tf.div(tf.reduce_sum(masked_losses), tf.reduce_sum(self.input_mask))
            self.loss += self.l2_penalty * self.l2_loss

            # run the forward pass to get scores
            self.unflat_no_dropout_scores = self.forward(self.input_x1, self.input_x2, self.max_seq_len,
                                                         1.0, 1.0, 1.0)

            drop_loss = tf.nn.l2_loss(tf.sub(self.unflat_scores, self.unflat_no_dropout_scores))
            self.loss += self.drop_penalty * drop_loss

        # Accuracy
        with tf.name_scope("predictions"):
            if viterbi:
                self.predictions = self.unflat_scores
            else:
                self.predictions = tf.argmax(self.unflat_scores, 2)

    def forward(self, input_x1, input_x2, max_seq_len, hidden_dropout_keep_prob,
                input_dropout_keep_prob, middle_dropout_keep_prob, reuse=True):
        word_embeddings = tf.nn.embedding_lookup(self.w_e, input_x1)

        with tf.variable_scope("forward", reuse=reuse):
            input_list = [word_embeddings]
            input_size = self.embedding_size
            if self.use_characters:
                input_list.append(self.char_embeddings)
                input_size += self.char_size
            if self.use_shape:
                shape_embeddings_shape = (self.shape_domain_size - 1, self.shape_size)
                w_s = tf_utils.initialize_embeddings(shape_embeddings_shape, name="w_s")
                shape_embeddings = tf.nn.embedding_lookup(w_s, input_x2)
                input_list.append(shape_embeddings)
                input_size += self.shape_size

            if self.use_geometric_feats:
                # TODO: add other features to input list, concat them to end of input_feats
                # todo this is the wrong shape to be concatenated
                # it's giving some issue with the typing, so I'm just casting everythint ot be the same
                input_list.append(tf.cast(self.widths, tf.float32))
                input_list.append(tf.cast(self.heights, tf.float32))
                input_list.append(tf.cast(self.wh_ratios, tf.float32))
                input_list.append(tf.cast(self.x_coords, tf.float32))
                input_list.append(tf.cast(self.y_coords, tf.float32))
                input_list.append(tf.cast(self.pages, tf.float32))
                input_list.append(tf.cast(self.lines, tf.float32))
                input_list.append(tf.cast(self.zones, tf.float32))
                input_size += 8

            # print(input.get_shape())
            # (w, h) = self.widths.get_shape()
            # print(tf.reshape(self.widths, (w, h, 1)).get_shape())

            input_feats = tf.concat(2, input_list)

            print(input_feats.get_shape())

            # self.input_feats_expanded = tf.expand_dims(self.input_feats, 1)
            input_feats_expanded_drop = tf.nn.dropout(input_feats, input_dropout_keep_prob)

            total_output_width = 2*self.hidden_dim

            with tf.name_scope("lstm"):
                # selected_col_embeddings = tf.nn.embedding_lookup(token_embeddings, self.token_batch)
                # todo change this to a normal dynamic rnn with lsmt cell
                fwd_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
                bwd_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
                lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell, cell_bw=bwd_cell, dtype=tf.float32,
                                                                 inputs=input_feats_expanded_drop,
                                                                 # inputs = input_feats,
                                                                 parallel_iterations=50,
                                                                 sequence_length=self.flat_sequence_lengths)
                hidden_outputs = tf.concat(2, lstm_outputs)

            # concatenate the results of the forward and backward cells
            h_concat_flat = tf.reshape(hidden_outputs, [-1, total_output_width])

            # Add dropout
            with tf.name_scope("middle_dropout"):
                h_drop = tf.nn.dropout(h_concat_flat, middle_dropout_keep_prob)

            # second projection
            with tf.name_scope("tanh_proj"):
                w_tanh = tf_utils.initialize_weights([total_output_width, self.hidden_dim], "w_tanh", init_type="xavier")
                b_tanh = tf.get_variable(initializer=tf.constant(0.01, shape=[self.hidden_dim]), name="b_tanh")
                self.l2_loss += tf.nn.l2_loss(w_tanh)
                self.l2_loss += tf.nn.l2_loss(b_tanh)
                h2_concat_flat = tf.nn.xw_plus_b(h_drop, w_tanh, b_tanh, name="h2_tanh")
                h2_tanh = tf_utils.apply_nonlinearity(h2_concat_flat, self.nonlinearity)

            # Add dropout
            with tf.name_scope("hidden_dropout"):
                h2_drop = tf.nn.dropout(h2_tanh, hidden_dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                w_o = tf_utils.initialize_weights([self.hidden_dim, self.num_classes], "w_o", init_type="xavier")
                b_o = tf.get_variable(initializer=tf.constant(0.01, shape=[self.num_classes]), name="b_o")
                self.l2_loss += tf.nn.l2_loss(w_o)
                self.l2_loss += tf.nn.l2_loss(b_o)
                scores = tf.nn.xw_plus_b(h2_drop, w_o, b_o, name="scores")
                unflat_scores = tf.reshape(scores, tf.pack([self.batch_size, max_seq_len, self.num_classes]))
        return unflat_scores

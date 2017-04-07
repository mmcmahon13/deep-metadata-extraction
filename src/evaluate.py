from __future__ import division
from __future__ import print_function
import sys
import time
import tensorflow as tf
import numpy as np
from models.batch_utils import SeqBatcher, Batcher
from models.bilstm import BiLSTM
from models.bilstm_char import BiLSTMChar
from models.cnn_char import CNNChar
import json
import tf_utils

FLAGS = tf.app.flags.FLAGS

# run the model on dev/test data and make predictions
# TODO untested
# def evaluate(sess, model, char_embedding_model, eval_batches, extra_text=""):
#     print(extra_text)
#     sys.stdout.flush()
#     predictions = []
#     for b, (eval_label_batch, eval_token_batch, eval_shape_batch, eval_char_batch, eval_seq_len_batch, eval_tok_len_batch,
#             eval_width_batch, eval_height_batch, eval_wh_ratio_batch, eval_x_coord_batch, eval_y_coord_batch,
#             eval_page_id_batch, eval_line_ids_batch, eval_zone_id_batch, eval_mask_batch) in enumerate(eval_batches):
#         batch_size, batch_seq_len = eval_token_batch.shape
#
#         char_lens = np.sum(eval_tok_len_batch, axis=1)
#         max_char_len = np.max(eval_tok_len_batch)
#         eval_padded_char_batch = np.zeros((batch_size, max_char_len * batch_seq_len))
#         for b in range(batch_size):
#             char_indices = [item for sublist in [range(i * max_char_len, i * max_char_len + d) for i, d in
#                                                  enumerate(eval_tok_len_batch[b])] for item in sublist]
#             eval_padded_char_batch[b, char_indices] = eval_char_batch[b][:char_lens[b]]
#
#         char_embedding_feeds = {} if FLAGS.char_dim == 0 else {
#             char_embedding_model.input_chars: eval_padded_char_batch,
#             char_embedding_model.batch_size: batch_size,
#             char_embedding_model.max_seq_len: batch_seq_len,
#             char_embedding_model.token_lengths: eval_tok_len_batch,
#             char_embedding_model.max_tok_len: max_char_len,
#             char_embedding_model.input_dropout_keep_prob: FLAGS.char_input_dropout
#         }
#
#         # print(eval_height_batch.get_shape())
#         # sys.stdout.flush()
#
#         # todo need to reshape these to add a third dimension
#         basic_feeds = {
#             model.input_x1: eval_token_batch,
#             model.input_x2: eval_shape_batch,
#             model.input_y: eval_label_batch,
#             model.input_mask: eval_mask_batch,
#             model.max_seq_len: batch_seq_len,
#             model.batch_size: batch_size,
#             model.sequence_lengths: eval_seq_len_batch,
#             model.widths: eval_width_batch,
#             model.heights: eval_height_batch,
#             model.wh_ratios: eval_wh_ratio_batch,
#             model.x_coords: eval_x_coord_batch,
#             model.y_coords: eval_y_coord_batch,
#             model.pages: eval_page_id_batch,
#             model.lines: eval_line_ids_batch,
#             model.zones: eval_zone_id_batch
#         }
#
#         basic_feeds.update(char_embedding_feeds)
#         total_feeds = basic_feeds.copy()
#
#         preds, scores = sess.run([model.predictions, model.unflat_scores], feed_dict=total_feeds)
#         predictions.append(preds)
#
#     return predictions

def print_training_error(num_examples, start_time, epoch_loss, step):
    print("%20d examples at %5.2f examples/sec. Error: %5.5f" %
          (num_examples, num_examples / (time.time() - start_time), (epoch_loss / step)))
    sys.stdout.flush()


# def f1_eval():
#     f1_micro, precision = evaluation.segment_eval(eval_batches, predictions, type_set, type_int_int_map,
#                                                   labels_id_str_map, vocab_id_str_map,
#                                                   outside_idx=map(
#                                                       lambda t: type_set[t] if t in type_set else type_set["O"],
#                                                       outside_set),
#                                                   pad_width=pad_width, start_end=FLAGS.start_end,
#                                                   extra_text="Segment evaluation %s:" % extra_text)
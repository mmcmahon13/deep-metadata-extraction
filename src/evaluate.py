from __future__ import division
from __future__ import print_function
import sys
import time

import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from models.batch_utils import SeqBatcher, Batcher
from models.bilstm import BiLSTM
from models.bilstm_char import BiLSTMChar
from models.cnn_char import CNNChar
import json
import tf_utils

FLAGS = tf.app.flags.FLAGS

# run the model on dev/test data and make predictions
# TODO alter this to return confusion matrix or save predictions of best model?
# also alter the preprocessing to save the document id so we can compare gold labels to predicted labels for a given doc
# encode in sequence an id that marks where it occurs on the page
def run_evaluation(sess, model, char_embedding_model, eval_batches, labels_str_id_map, labels_id_str_map, extra_text=""):
    print(extra_text)
    sys.stdout.flush()
    predictions = []
    labels = []
    for b, batch in enumerate(eval_batches):
        # print("Batch: ", batch)
        # sys.stdout.flush()
        (eval_label_batch, eval_token_batch, eval_shape_batch, eval_char_batch, eval_seq_len_batch, eval_tok_len_batch,
         eval_width_batch, eval_height_batch, eval_wh_ratio_batch, eval_x_coord_batch, eval_y_coord_batch,
         eval_page_id_batch, eval_line_id_batch, eval_zone_id_batch,
         eval_place_scores_batch, eval_department_scores_batch, eval_university_scores_batch, eval_person_scores_batch,
         eval_mask_batch) = batch
        batch_size, batch_seq_len = eval_token_batch.shape

        # reshape the features to be 3d tensors with 3rd dim = 1 (batch size) x (seq_len) x (1)
        # print("Reshaping features....")
        eval_width_batch = np.expand_dims(eval_width_batch, axis=2)
        eval_height_batch = np.expand_dims(eval_height_batch, axis=2)
        eval_wh_ratio_batch = np.expand_dims(eval_wh_ratio_batch, axis=2)
        eval_x_coord_batch = np.expand_dims(eval_x_coord_batch, axis=2)
        eval_y_coord_batch = np.expand_dims(eval_y_coord_batch, axis=2)
        eval_page_id_batch = np.expand_dims(eval_page_id_batch, axis=2)
        eval_line_id_batch = np.expand_dims(eval_line_id_batch, axis=2)
        eval_zone_id_batch = np.expand_dims(eval_zone_id_batch, axis=2)
        # eval_place_scores_batch = np.expand_dims(eval_place_scores_batch, axis=2)
        # eval_department_scores_batch = np.expand_dims(eval_department_scores_batch, axis=2)
        # eval_university_scores_batch = np.expand_dims(eval_university_scores_batch, axis=2)
        # eval_person_scores_batch = np.expand_dims(eval_person_scores_batch, axis=2)

        char_lens = np.sum(eval_tok_len_batch, axis=1)
        max_char_len = np.max(eval_tok_len_batch)
        eval_padded_char_batch = np.zeros((batch_size, max_char_len * batch_seq_len))
        for b in range(batch_size):
            char_indices = [item for sublist in [range(i * max_char_len, i * max_char_len + d) for i, d in
                                                 enumerate(eval_tok_len_batch[b])] for item in sublist]
            eval_padded_char_batch[b, char_indices] = eval_char_batch[b][:char_lens[b]]

        char_embedding_feeds = {} if FLAGS.char_dim == 0 else {
            char_embedding_model.input_chars: eval_padded_char_batch,
            char_embedding_model.batch_size: batch_size,
            char_embedding_model.max_seq_len: batch_seq_len,
            char_embedding_model.token_lengths: eval_tok_len_batch,
            char_embedding_model.max_tok_len: max_char_len,
            char_embedding_model.input_dropout_keep_prob: FLAGS.char_input_dropout
        }

        # print(eval_height_batch.get_shape())
        # sys.stdout.flush()

        # todo need to reshape these to add a third dimension
        basic_feeds = {
            model.input_x1: eval_token_batch,
            model.input_x2: eval_shape_batch,
            model.input_y: eval_label_batch,
            model.input_mask: eval_mask_batch,
            model.max_seq_len: batch_seq_len,
            model.batch_size: batch_size,
            model.sequence_lengths: eval_seq_len_batch,
            model.widths: eval_width_batch,
            model.heights: eval_height_batch,
            model.wh_ratios: eval_wh_ratio_batch,
            model.x_coords: eval_x_coord_batch,
            model.y_coords: eval_y_coord_batch,
            model.pages: eval_page_id_batch,
            model.lines: eval_line_id_batch,
            model.zones: eval_zone_id_batch
        }

        geometric_feats_feeds = {
            model.widths: eval_width_batch,
            model.heights: eval_height_batch,
            model.wh_ratios: eval_wh_ratio_batch,
            model.x_coords: eval_x_coord_batch,
            model.y_coords: eval_y_coord_batch,
            model.pages: eval_page_id_batch,
            model.lines: eval_line_id_batch,
            model.zones: eval_zone_id_batch,
        }

        lexicon_feats_feeds = {
            model.place_scores: eval_place_scores_batch,
            model.department_scores: eval_department_scores_batch,
            model.university_scores: eval_university_scores_batch,
            model.person_scores: eval_person_scores_batch
        }


        basic_feeds.update(char_embedding_feeds)

        if FLAGS.use_geometric_feats:
            basic_feeds.update(geometric_feats_feeds)

        if FLAGS.use_lexicons:
            basic_feeds.update(lexicon_feats_feeds)

        total_feeds = basic_feeds.copy()

        # predictions for one batch
        preds, scores = sess.run([model.predictions, model.unflat_scores], feed_dict=total_feeds)
        # tf.Print(preds, [tf.shape(preds)], message='predictions shape:')
        predictions.append(preds)
        # print(preds.shape)
        labels.append(eval_label_batch)
        # print(eval_label_batch.shape)
        # sys.stdout.flush()

    # print("starting batch evaluation")
    flat_preds = np.concatenate([p.flatten() for p in predictions])
    flat_labels = np.concatenate([l.flatten() for l in labels])

    flat_preds = np.array([labels_id_str_map[p] for p in flat_preds])
    flat_labels = np.array([labels_id_str_map[l] for l in flat_labels])
    # print(len(flat_preds))
    # print(len(flat_labels))
    # print(flat_preds[0])
    sys.stdout.flush()

    # print(labels_str_id_map.keys())
    # sys.stdout.flush()

    tag_set = set([l.split('-')[-1] for l in labels_str_id_map.keys()])

    tag_level_metrics = compute_f1_score(flat_labels, flat_preds, tag_set)
    accuracy = sum(flat_preds == flat_labels) * 1. / len(flat_labels)

    # TODO also print out micro/macro?
    w_f1 = f1_score(flat_labels, flat_preds, average='weighted')
    micro_f1 = f1_score(flat_labels, flat_preds, average='micro')
    macro_f1 = f1_score(flat_labels, flat_preds, average='macro')
    print('Weighted F1: %f' % w_f1)
    print('Micro F1: %f' % micro_f1)
    print('Macro F1: %f' % macro_f1)
    print('Accuracy: %f' % accuracy)
    print()

    for tag in tag_level_metrics:
        print('Precision, Recall, F1 for ' + str(tag) + ': ' + str(tag_level_metrics[tag][0]) + ', ' + str(
            tag_level_metrics[tag][1]) + ', ' + str(tag_level_metrics[tag][2]))
    sys.stdout.flush()
        # f1_micro, precision = evaluation.segment_eval(eval_batches, predictions, type_set, type_int_int_map,
        #                                    labels_id_str_map, vocab_id_str_map,
        #                                    outside_idx=map(lambda t: type_set[t] if t in type_set else type_set["O"], outside_set),
        #                                    pad_width=pad_width, start_end=FLAGS.start_end,
        #                                    extra_text="Segment evaluation %s:" % extra_text)
        # # evaluation.token_eval(dev_batches, predictions, type_set, type_int_int_map, outside_idx=type_set["O"],
        # #                       pad_width=pad_width, extra_text="Token evaluation %s:" % extra_text)
        # # evaluation.boundary_eval(eval_batches, predictions, bilou_set, bilou_int_int_map,
        # #                          outside_idx=bilou_set["O"], pad_width=pad_width,
        # #                          extra_text="Boundary evaluation %s: " % extra_text)
        # # print("done with batch evaluation")
        # return f1_micro, precision
    return w_f1, accuracy, flat_preds, flat_labels

def compute_f1_score(ytrue, ypred, tag_set):
    # this is direct from the Meta example script Shankar sent
    tag_level_metrics = dict()

    # get the types without the BIO
    ytrue = np.array([y.split('-')[1] if y != 'O' else y for y in ytrue])
    ypred = np.array([y.split('-')[1] if y != 'O' else y for y in ypred])

    for tag in tag_set:
        ids = np.where(ytrue == tag)[0]
        if len(ids) == 0: continue
        yt = np.zeros(len(ytrue))
        yp = np.zeros(len(ytrue))
        yt[ids] = 1
        yp[np.where(ypred == tag)] = 1

        tp = np.dot(yp, yt)
        fn = len(ids) - tp
        fp = sum(yp[np.setdiff1d(np.arange(len(ytrue)), ids)])

        if tp == 0:
            tag_level_metrics[tag] = (0, 0, 0)
        else:
            p = tp * 1. / (tp + fp)
            r = tp * 1. / (tp + fn)
            f1 = 2. * p * r / (p + r)
            tag_level_metrics[tag] = (p, r, f1)

    return tag_level_metrics

def print_training_error(num_examples, start_time, epoch_loss, step):
    print("%20d examples at %5.2f examples/sec. Error: %5.5f" %
          (num_examples, num_examples / (time.time() - start_time), (epoch_loss / step)))
    sys.stdout.flush()
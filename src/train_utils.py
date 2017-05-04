from __future__ import division

import sys
import tensorflow as tf
import numpy as np
import tf_utils

FLAGS = tf.app.flags.FLAGS


def sample_pad_size():
    return np.random.randint(1, FLAGS.max_additional_pad) if FLAGS.max_additional_pad > 0 else 0

def create_type_maps(labels_str_id_map):
    type_int_int_map = {}
    bilou_int_int_map = {}
    bilou_set = {}
    type_set = {}
    outside_set = ["O", "<PAD>", "<S>", "</S>", "<ZERO>"]

    # create sets of both type and bilou-encoded labels
    for label, id in labels_str_id_map.items():
        label_type = label if label in outside_set else label[2:]
        label_bilou = label[0]
        if label_type not in type_set:
            type_set[label_type] = len(type_set)
        if label_bilou not in bilou_set:
            bilou_set[label_bilou] = len(bilou_set)
        type_int_int_map[id] = type_set[label_type]
        bilou_int_int_map[id] = bilou_set[label_bilou]

    # manually add O key if it's not in type_set already
    if not type_set.has_key("O"):
        type_set["O"] = len(type_set)

    type_int_str_map = {a: b for b, a in type_set.items()}
    bilou_int_str_map = {a: b for b, a in bilou_set.items()}
    num_types = len(type_set)
    num_bilou = len(bilou_set)
    print(type_set)

    return type_int_int_map, bilou_int_int_map, type_set, bilou_set

# load the maps created during preprocessing
def load_intmaps(train_dir):
    print("Loading vocabulary maps...")
    sys.stdout.flush()
    with open(dir + '/label.txt', 'r') as f:
        labels_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        labels_id_str_map = {i: s for s, i in labels_str_id_map.items()}
        labels_size = len(labels_id_str_map)
    with open(dir + '/token.txt', 'r') as f:
        vocab_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        vocab_id_str_map = {i: s for s, i in vocab_str_id_map.items()}
        vocab_size = len(vocab_id_str_map)
    with open(dir + '/shape.txt', 'r') as f:
        shape_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        shape_id_str_map = {i: s for s, i in shape_str_id_map.items()}
        shape_domain_size = len(shape_id_str_map)
    with open(dir + '/char.txt', 'r') as f:
        char_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        char_id_str_map = {i: s for s, i in char_str_id_map.items()}
        char_domain_size = len(char_id_str_map)
    print("Loaded.")
    sys.stdout.flush()
    return labels_str_id_map, labels_id_str_map, vocab_str_id_map, vocab_id_str_map, shape_str_id_map, shape_id_str_map, char_str_id_map, char_id_str_map

# load the word embeddings
def load_embeddings(vocab_str_id_map):
    print("Loading embeddings...")
    sys.stdout.flush()
    vocab_size = len(vocab_str_id_map)
    # load embeddings, if given; initialize in range [-.01, .01]
    embeddings_shape = (vocab_size - 1, FLAGS.embed_dim)
    embeddings = tf_utils.embedding_values(embeddings_shape, old=True)
    embeddings_used = 0
    if FLAGS.embeddings != '':
        with open(FLAGS.embeddings, 'r') as f:
            for line in f.readlines():
                split_line = line.strip().split(" ")
                word = split_line[0]
                embedding = split_line[1:]
                # print("word: %s" % word)
                # print("embedding: %s" % embedding)
                # shift by -1 because we are going to add a 0 constant vector for the padding later?
                if word in vocab_str_id_map and word != "<PAD>" and word != "<OOV>":
                    embeddings_used += 1
                    embeddings[vocab_str_id_map[word] - 1] = map(float, embedding)
                elif word.lower() in vocab_str_id_map and word != "<OOV>":
                    embeddings_used += 1
                    embeddings[vocab_str_id_map[word.lower()] - 1] = map(float, embedding)
                else:
                    pass
                    # print("out of vocab: %s" % word)
    # TODO i don't really understand how the embeddings are used in the preprocessing?
    print("Loaded %d/%d embeddings (%2.2f%% coverage)" % (
    embeddings_used, vocab_size, embeddings_used / vocab_size * 100))
    sys.stdout.flush()
    return embeddings

# print out the number of trainable params in the model
def get_trainable_params():
    total_parameters=0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(variable.name, shape)
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total trainable parameters: %d" % (total_parameters))
    sys.stdout.flush()

def load_batches(sess, train_batcher, train_eval_batcher, dev_batcher, pad_width=0):

    dev_batches = []
    # load all the dev batches into memory
    done = False
    print("Loading dev batches...")
    sys.stdout.flush()
    num_batches = 0
    num_dev_examples = 0
    while not done:
        try:
            dev_batch = sess.run(dev_batcher.next_batch_op)
            print("loaded dev batch %d" % num_batches)
            sys.stdout.flush()
            dev_label_batch, dev_token_batch, dev_shape_batch, dev_char_batch, dev_seq_len_batch, dev_tok_len_batch, \
            dev_width_batch, dev_height_batch, dev_wh_ratio_batch, dev_x_coord_batch, dev_y_coord_batch, \
            dev_page_id_batch, dev_line_id_batch, dev_zone_id_batch, \
            dev_place_scores_batch, dev_department_scores_batch, dev_university_scores_batch, dev_person_scores_batch = dev_batch
            print("batch length: %d" % len(dev_seq_len_batch))
            sys.stdout.flush()
            num_dev_examples += len(dev_seq_len_batch)
            mask_batch = np.zeros(dev_token_batch.shape)
            for i, seq_lens in enumerate(dev_seq_len_batch):
                # print("creating mask for batch %d" % i)
                sys.stdout.flush()
                # todo is this masking correctly? why are we adding the seq_len to start each time?
                start = pad_width
                for seq_len in seq_lens:
                    mask_batch[i, start:start + seq_len] = 1
                    # + (2 if FLAGS.start_end else 1) * pad_width
                    start += seq_len
            dev_batches.append((dev_label_batch, dev_token_batch, dev_shape_batch, dev_char_batch, dev_seq_len_batch,
                                dev_tok_len_batch, dev_width_batch, dev_height_batch, dev_wh_ratio_batch, dev_x_coord_batch,
                                dev_y_coord_batch, dev_page_id_batch, dev_line_id_batch, dev_zone_id_batch,
                                dev_place_scores_batch, dev_department_scores_batch, dev_university_scores_batch,
                                dev_person_scores_batch, mask_batch))
            num_batches += 1
        except:
            # print("Error loading dev batches")
            done = True

    print("%d dev batches loaded." % len(dev_batches))
    print()
    sys.stdout.flush()

    print("Loading train batches...")
    sys.stdout.flush()
    num_batches = 0
    num_train_examples = 0
    train_batches = []
    if FLAGS.train_eval:
        # load all the train batches into memory if we want to report training accuracy
        done = False
        while not done:
            try:
                train_batch = sess.run(train_eval_batcher.next_batch_op)
                print("loaded train batch %d" % num_batches)
                sys.stdout.flush()
                train_label_batch, train_token_batch, train_shape_batch, train_char_batch, train_seq_len_batch, train_tok_len_batch,\
                train_width_batch, train_height_batch, train_wh_ratio_batch, train_x_coord_batch, train_y_coord_batch, \
                train_page_id_batch, train_line_id_batch, train_zone_id_batch, \
                train_place_scores_batch, train_department_scores_batch, train_university_scores_batch, train_person_scores_batch = train_batch
                mask_batch = np.zeros(train_token_batch.shape)
                print("batch length: %d" % len(train_seq_len_batch))
                sys.stdout.flush()
                num_train_examples += len(train_seq_len_batch)
                for i, seq_lens in enumerate(train_seq_len_batch):
                    start = pad_width
                    for seq_len in seq_lens:
                        mask_batch[i, start:start + seq_len] = 1
                        # + (2 if FLAGS.start_end else 1) * pad_width
                        start += seq_len
                train_batches.append((train_label_batch, train_token_batch, train_shape_batch, train_char_batch, train_seq_len_batch, train_tok_len_batch,\
                                        train_width_batch, train_height_batch, train_wh_ratio_batch, train_x_coord_batch, train_y_coord_batch, \
                                        train_page_id_batch, train_line_id_batch, train_zone_id_batch, \
                                        train_place_scores_batch, train_department_scores_batch, \
                                        train_university_scores_batch, train_person_scores_batch, mask_batch))
                num_batches += 1
            except Exception as e:
                # print("Error loading train batches")
                done = True
    else:
        done = False
        while not done:
            try:
                # determine the number of train examples but don't actually load them
                train_batch = sess.run(train_eval_batcher.next_batch_op)
                train_label_batch, train_token_batch, train_shape_batch, train_char_batch, train_seq_len_batch, train_tok_len_batch, \
                train_width_batch, train_height_batch, train_wh_ratio_batch, train_x_coord_batch, train_y_coord_batch, \
                train_page_id_batch, train_line_id_batch, train_zone_id_batch, \
                train_place_scores_batch, train_department_scores_batch, train_university_scores_batch, train_person_scores_batch = train_batch
                mask_batch = np.zeros(train_token_batch.shape)
                print("batch length: %d" % len(train_seq_len_batch))
                sys.stdout.flush()
                num_train_examples += len(train_seq_len_batch)
            except Exception as e:
                # print("Error loading train batches")
                done = True

    if FLAGS.memmap_train:
        train_batcher.load_and_bucket_data(sess)
    print("%d train batches loaded." % len(train_batches))
    print()
    sys.stdout.flush()

    return dev_batches, train_batches, num_dev_examples, num_train_examples
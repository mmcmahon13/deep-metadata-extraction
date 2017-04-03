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

# FLAGS
# data directories
tf.app.flags.DEFINE_string('train_dir', '', 'directory containing preprocessed training data')
tf.app.flags.DEFINE_string('dev_dir', '', 'directory containing preprocessed dev data')
tf.app.flags.DEFINE_string('test_dir', '', 'directory containing preprocessed test data')

# embeddings and dimensions
tf.app.flags.DEFINE_string('embeddings', '', 'path to embeddings file')
tf.app.flags.DEFINE_integer('embed_dim', 200, 'dimensions of the words embeddings')

# character embeddings
tf.app.flags.DEFINE_integer('char_dim', 0, 'character embedding dimension') # set to 25?
tf.app.flags.DEFINE_integer('char_tok_dim', 0, 'character token embedding dimension')
tf.app.flags.DEFINE_string('char_model', 'lstm', 'character embedding model (lstm, cnn)')

tf.app.flags.DEFINE_integer('shape_dim', 5, 'shape embedding dimension')

# lstm layer dimensions
tf.app.flags.DEFINE_integer('lstm_dim', 2048, 'lstm internal dimension')

# training
tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size')
tf.app.flags.DEFINE_boolean('train_eval', False, 'whether to report train accuracy')
tf.app.flags.DEFINE_boolean('memmap_train', False, 'whether to load all training examples into memory')
tf.app.flags.DEFINE_string('master', '', 'use for Supervisor')
tf.app.flags.DEFINE_integer('max_epochs', 100, 'train for this many epochs')
tf.app.flags.DEFINE_boolean('until_convergence', False, 'whether to run until convergence')
tf.app.flags.DEFINE_boolean('use_geometric_feats', False, 'whether to use the geometric features')

# hyperparams
tf.app.flags.DEFINE_string('nonlinearity', 'relu', 'nonlinearity function to use (tanh, sigmoid, relu)')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('l2', 0.0, 'l2 penalty')
tf.app.flags.DEFINE_float('beta1', 0.9, 'beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'beta2')
tf.app.flags.DEFINE_float('epsilon', 1e-8, 'epsilon')

# dropouts
tf.app.flags.DEFINE_float('hidden_dropout', .75, 'hidden layer dropout rate')
tf.app.flags.DEFINE_float('hidden2_dropout', .75, 'hidden layer 2 dropout rate')
tf.app.flags.DEFINE_float('input2_dropout', .75, 'input layer 2 dropout rate')

tf.app.flags.DEFINE_float('input_dropout', 1.0, 'input layer (word embedding) dropout rate')
tf.app.flags.DEFINE_float('middle_dropout', 1.0, 'middle layer dropout rate')
tf.app.flags.DEFINE_float('word_dropout', 1.0, 'whole-word (-> oov) dropout rate')

# penalties
tf.app.flags.DEFINE_float('regularize_drop_penalty', 0.0, 'penalty for dropout regularization')

# saving and loading models
tf.app.flags.DEFINE_string('model_dir', '', 'save model to this dir (if empty do not save)')
tf.app.flags.DEFINE_string('load_dir', '', 'load model from this dir (if empty do not load)')

FLAGS = tf.app.flags.FLAGS


def sample_pad_size():
    return np.random.randint(1, FLAGS.max_additional_pad) if FLAGS.max_additional_pad > 0 else 0

# load the maps created during preprocessing
def load_intmaps():
    print("Loading vocabulary maps...")
    sys.stdout.flush()
    with open(FLAGS.train_dir + '/label.txt', 'r') as f:
        labels_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        labels_id_str_map = {i: s for s, i in labels_str_id_map.items()}
        labels_size = len(labels_id_str_map)
    with open(FLAGS.train_dir + '/token.txt', 'r') as f:
        vocab_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        vocab_id_str_map = {i: s for s, i in vocab_str_id_map.items()}
        vocab_size = len(vocab_id_str_map)
    with open(FLAGS.train_dir + '/shape.txt', 'r') as f:
        shape_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        shape_id_str_map = {i: s for s, i in shape_str_id_map.items()}
        shape_domain_size = len(shape_id_str_map)
    with open(FLAGS.train_dir + '/char.txt', 'r') as f:
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
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total trainable parameters: %d" % (total_parameters))
    sys.stdout.flush()

# run the model on dev/test data and make predictions
# TODO untested
def evaluate(sess, model, char_embedding_model, eval_batches, extra_text=""):
    print(extra_text)
    sys.stdout.flush()
    predictions = []
    for b, (eval_label_batch, eval_token_batch, eval_shape_batch, eval_char_batch, eval_seq_len_batch, eval_tok_len_batch,
            eval_width_batch, eval_height_batch, eval_wh_ratio_batch, eval_x_coord_batch, eval_y_coord_batch,
            eval_page_id_batch, eval_line_ids_batch, eval_zone_id_batch, eval_mask_batch) in enumerate(eval_batches):
        batch_size, batch_seq_len = eval_token_batch.shape

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

        print(eval_height_batch.get_shape())
        sys.stdout.flush()

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
            model.lines: eval_line_ids_batch,
            model.zones: eval_zone_id_batch
        }

        basic_feeds.update(char_embedding_feeds)
        total_feeds = basic_feeds.copy()

        preds, scores = sess.run([model.predictions, model.unflat_scores], feed_dict=total_feeds)
        predictions.append(preds)

    return predictions

# TODO: change this to not load all batches into memory at once? or is it fine? will have to test with bigger files
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
            # print("loaded dev batch %d" % num_batches)
            sys.stdout.flush()
            dev_label_batch, dev_token_batch, dev_shape_batch, dev_char_batch, dev_seq_len_batch, dev_tok_len_batch, \
            dev_width_batch, dev_height_batch, dev_wh_ratio_batch, dev_x_coord_batch, dev_y_coord_batch, \
            dev_page_id_batch, dev_line_id_batch, dev_zone_id_batch = dev_batch
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
                                dev_y_coord_batch, dev_page_id_batch, dev_line_id_batch, dev_zone_id_batch, mask_batch))
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
                train_page_id_batch, train_line_id_batch, train_zone_id_batch = train_batch
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
                                        train_page_id_batch, train_line_id_batch, train_zone_id_batch))
                num_batches += 1
            except Exception as e:
                # print("Error loading train batches")
                done = True
    if FLAGS.memmap_train:
        train_batcher.load_and_bucket_data(sess)
    print("%d train batches loaded." % len(train_batches))
    print()
    sys.stdout.flush()

    return dev_batches, train_batches, num_dev_examples, num_train_examples


def train():
    # load preprocessed maps and embeddings
    labels_str_id_map, labels_id_str_map, vocab_str_id_map, vocab_id_str_map, \
    shape_str_id_map, shape_id_str_map, char_str_id_map, char_id_str_map = load_intmaps()
    embeddings = load_embeddings(vocab_str_id_map)

    labels_size = len(labels_str_id_map)
    char_domain_size = len(char_id_str_map)
    vocab_size = len(vocab_str_id_map)
    shape_domain_size = len(shape_id_str_map)

    with tf.Graph().as_default():
        train_batcher = Batcher(FLAGS.train_dir, FLAGS.batch_size) if FLAGS.memmap_train else SeqBatcher(FLAGS.train_dir,
                                                                                                   FLAGS.batch_size)
        dev_batcher = SeqBatcher(FLAGS.dev_dir, FLAGS.batch_size, num_buckets=0, num_epochs=1)

        train_eval_batcher = SeqBatcher(FLAGS.train_dir, FLAGS.batch_size, num_buckets=0, num_epochs=1)

        # create character embedding model and train char embeddings:
        # todo this is broken, fix it and add it in when I get the rest of the network running
        if FLAGS.char_dim > 0 and FLAGS.char_model == "lstm":
            print("creating and training character embeddings")
            char_embedding_model = BiLSTMChar(char_domain_size, FLAGS.char_dim, int(FLAGS.char_tok_dim / 2))
        # elif FLAGS.char_dim > 0 and FLAGS.char_model == "cnn":
        #     char_embedding_model = CNNChar(char_domain_size, FLAGS.char_dim, FLAGS.char_tok_dim, layers_map[0][1]['width'])
        else:
            char_embedding_model = None
        char_embeddings = char_embedding_model.outputs if char_embedding_model is not None else None

        # create BiLSTM model
        model = BiLSTM(
            num_classes=labels_size,
            vocab_size=vocab_size,
            shape_domain_size=shape_domain_size,
            char_domain_size=char_domain_size,
            char_size=FLAGS.char_dim,
            embedding_size=FLAGS.embed_dim,
            shape_size=FLAGS.shape_dim,
            nonlinearity=FLAGS.nonlinearity,
            viterbi=False, #viterbi=FLAGS.viterbi,
            hidden_dim=FLAGS.lstm_dim,
            char_embeddings=char_embeddings,
            embeddings=embeddings,
            use_geometric_feats=FLAGS.use_geometric_feats)

        # Define Training procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=FLAGS.beta1, beta2=FLAGS.beta2,
                                           epsilon=FLAGS.epsilon, name="optimizer")

        model_vars = [v for v in tf.all_variables() if 'context_agg' not in v.name]

        train_op = optimizer.minimize(model.loss, global_step=global_step, var_list=model_vars)

        print("model vars: %d" % len(model_vars))
        print(map(lambda v: v.name, model_vars))
        print()
        sys.stdout.flush()
        get_trainable_params()

        tf.initialize_all_variables()

        sv = tf.python.train.Supervisor(logdir=FLAGS.model_dir if FLAGS.model_dir != '' else None,
                                        global_step=global_step,
                                        saver=None,
                                        save_model_secs=0,
                                        save_summaries_secs=0
                                        )

        training_start_time = time.time()

        # create session
        with sv.managed_session(FLAGS.master, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            print("session created")
            sys.stdout.flush()
            # start queue runner threads
            threads = tf.train.start_queue_runners(sess=sess)

            # load batches
            print()
            dev_batches, train_batches, num_dev_examples, num_train_examples = load_batches(sess, train_batcher, train_eval_batcher, dev_batcher)

            # this seems wrong?
            # num_train_examples = 0
            # for b in train_batches:
            #     num_train_examples += len(b)
            #
            # num_dev_examples = 0
            # for b in dev_batches:
            #     num_dev_examples += len(b)
            #
            # print("num dev examples: %d" % num_dev_examples)

            log_every = int(max(100, num_train_examples / 5))

            # start the training loop
            print("Training on %d pages (%d examples)" % (num_train_examples, num_train_examples))
            sys.stdout.flush()
            start_time = time.time()
            train_batcher._step = 1.0
            converged = False
            examples = 0
            log_every_running = log_every
            epoch_loss = 0.0
            num_lower = 0
            training_iteration = 0
            speed_num = 0.0
            speed_denom = 0.0
            # todo we might want this to be false for finetuning or something?
            update_context = True
            while not sv.should_stop() and training_iteration < FLAGS.max_epochs and not (FLAGS.until_convergence and converged):
                print("training iteration %d" % training_iteration)
                sys.stdout.flush()
                training_iteration += 1
                # if examples >= num_train_examples:
                #     training_iteration += 1
                #     # print("iteration %d" % training_iteration)
                #     sys.stdout.flush()
                #
                #     if FLAGS.train_eval:
                #         evaluate(sess, train_batches, update_context, "TRAIN (iteration %d)" % training_iteration)
                #         sys.stdout.flush()
                #     print()
                #     # TODO implement evalaution metrics
                #     f1_micro, precision = evaluate(sess, dev_batches, update_context,
                #                                          "TEST (iteration %d)" % training_iteration)
                #     print("Avg training speed: %f examples/second" % (speed_num / speed_denom))
                #     sys.stdout.flush()


                # do training
                label_batch, token_batch, shape_batch, char_batch, seq_len_batch, tok_lengths_batch,\
                    widths_batch, heights_batch, wh_ratios_batch, x_coords_batch, y_coords_batch,\
                    page_ids_batch, line_ids_batch, zone_ids_batch = \
                    train_batcher.next_batch() if FLAGS.memmap_train else sess.run(train_batcher.next_batch_op)

                # check that shapes look correct
                # print("label_batch_shape: ", label_batch.shape)
                # print("token batch shape: ", token_batch.shape)
                # print("shape batch shape: ", shape_batch.shape)
                # print("char batch shape: ", char_batch.shape)
                print("seq_len batch shape: ", seq_len_batch.shape)
                # print("tok_len batch shape: ", tok_lengths_batch.shape)
                # print("widths_batch shape: ", widths_batch.shape)
                # print("heights_batch shape: ", heights_batch.shape)
                # print("ratios_batch shape: ", wh_ratios_batch.shape)
                # print("x_coords shape: ", x_coords_batch.shape)
                # print("y_coords shape: ", y_coords_batch.shape)
                # print("pages shape: ", page_ids_batch.shape)
                # print("lines shape: ", line_ids_batch.shape)
                # print("zones shape: ", zone_ids_batch.shape)
                #
                # print("Max sequence length in batch: %d" % np.max(seq_len_batch))
                sys.stdout.flush()

                # reshape the features to be 3d tensors with 3rd dim = 1 (batch size) x (seq_len) x (1)
                print("Reshaping features....")
                widths_batch = np.expand_dims(widths_batch, axis=2)
                heights_batch = np.expand_dims(heights_batch, axis=2)
                wh_ratios_batch = np.expand_dims(wh_ratios_batch, axis=2)
                x_coords_batch = np.expand_dims(x_coords_batch, axis=2)
                y_coords_batch = np.expand_dims(y_coords_batch, axis=2)
                page_ids_batch = np.expand_dims(page_ids_batch, axis=2)
                line_ids_batch = np.expand_dims(line_ids_batch, axis=2)
                zone_ids_batch = np.expand_dims(zone_ids_batch, axis=2)

                # make mask out of seq lens
                batch_size, batch_seq_len = token_batch.shape

                # print(batch_seq_len)

                # pad the character batch?
                char_lens = np.sum(tok_lengths_batch, axis=1)
                max_char_len = np.max(tok_lengths_batch)
                padded_char_batch = np.zeros((batch_size, max_char_len * batch_seq_len))
                for b in range(batch_size):
                    char_indices = [item for sublist in [range(i * max_char_len, i * max_char_len + d) for i, d in
                                                         enumerate(tok_lengths_batch[b])] for item in sublist]
                    padded_char_batch[b, char_indices] = char_batch[b][:char_lens[b]]

                num_sentences_batch = np.sum(seq_len_batch != 0, axis=1)

                # print(seq_len_batch)
                # print(num_sentences_batch)
                pad_width = 0

                # create masks for each example based on sequence lengths
                mask_batch = np.zeros((batch_size, batch_seq_len))
                for i, seq_lens in enumerate(seq_len_batch):
                    start = pad_width
                    for seq_len in seq_lens:
                        mask_batch[i, start:start + seq_len] = 1
                        start += seq_len #+ (2 if FLAGS.start_end else 1) * pad_width
                examples += batch_size

                # print(batch_seq_len)
                # print(tok_lengths_batch.shape)
                # print(tok_lengths_batch)
                # print(np.reshape(tok_lengths_batch, (batch_size*batch_seq_len)))
                # print(padded_char_batch.shape)
                # print(np.reshape(padded_char_batch, (batch_size*batch_seq_len, max_char_len)))

                # print("input_x1", token_batch)
                # print("input_x1_sample_pad", input_x1_sample_pad)

                # apply word dropout
                # create word dropout mask
                word_probs = np.random.random(token_batch.shape)
                drop_indices = np.where((word_probs > FLAGS.word_dropout)) #& (token_batch != vocab_str_id_map["<PAD>"]))
                token_batch[drop_indices[0], drop_indices[1]] = vocab_str_id_map["<OOV>"]

                # TODO what is going on here
                # # sample padding
                # # sample an amount of padding to add for each sentence
                #
                # max_sampled_seq_len = batch_seq_len + (np.max(num_sentences_batch) + 1) * FLAGS.max_additional_pad
                # input_x1_sample_pad = np.empty((batch_size, max_sampled_seq_len))
                # input_x2_sample_pad = np.empty((batch_size, max_sampled_seq_len))
                # # input_x3_sample_pad = np.empty((batch_size, max_sampled_seq_len))
                # input_mask_sample_pad = np.zeros((batch_size, max_sampled_seq_len))
                # if FLAGS.regularize_pad_penalty != 0.0:
                #     input_x1_sample_pad.fill(vocab_str_id_map["<PAD>"])
                #     input_x2_sample_pad.fill(shape_str_id_map["<PAD>"])
                #     # input_x3_sample_pad.fill(char_str_id_map["<PAD>"])
                #
                #     for i, seq_lens in enumerate(seq_len_batch):
                #         pad_start = sample_pad_size()
                #         actual_start = pad_width
                #         for seq_len in seq_lens:
                #             input_x1_sample_pad[i, pad_start:pad_start + seq_len] = token_batch[i,
                #                                                                     actual_start:actual_start + seq_len]
                #             input_x2_sample_pad[i, pad_start:pad_start + seq_len] = shape_batch[i,
                #                                                                     actual_start:actual_start + seq_len]
                #             # input_x3_sample_pad[i, pad_start:pad_start+seq_len] = char_batch[i, actual_start:actual_start+seq_len]
                #             input_mask_sample_pad[i, pad_start:pad_start + seq_len] = 1
                #             sampled_pad_size = sample_pad_size()
                #             pad_start += seq_len + sampled_pad_size
                #             actual_start += seq_len + (2 if FLAGS.start_end else 1) * pad_width

                char_embedding_feeds = {} if FLAGS.char_dim == 0 else {
                    char_embedding_model.input_chars: padded_char_batch,
                    char_embedding_model.batch_size: batch_size,
                    char_embedding_model.max_seq_len: batch_seq_len,
                    char_embedding_model.token_lengths: tok_lengths_batch,
                    char_embedding_model.max_tok_len: max_char_len
                }

                lstm_feed = {
                    model.input_x1: token_batch,
                    model.input_x2: shape_batch,
                    model.input_y: label_batch,
                    model.input_mask: mask_batch,
                    model.sequence_lengths: seq_len_batch,
                    model.max_seq_len: batch_seq_len,
                    model.batch_size: batch_size,
                    model.hidden_dropout_keep_prob: FLAGS.hidden_dropout,
                    model.middle_dropout_keep_prob: FLAGS.middle_dropout,
                    model.input_dropout_keep_prob: FLAGS.input_dropout,
                    model.l2_penalty: FLAGS.l2,
                    model.drop_penalty: FLAGS.regularize_drop_penalty
                }


                geometric_feats_feeds = {
                    model.widths: widths_batch,
                    model.heights: heights_batch,
                    model.wh_ratios: wh_ratios_batch,
                    model.x_coords: x_coords_batch,
                    model.y_coords: y_coords_batch,
                    model.pages: page_ids_batch,
                    model.lines: line_ids_batch,
                    model.zones: zone_ids_batch,
                }

                lstm_feed.update(char_embedding_feeds)

                if FLAGS.use_geometric_feats:
                    lstm_feed.update(geometric_feats_feeds)

                print("Running training op:")
                sys.stdout.flush()
                _, loss = sess.run([train_op, model.loss], feed_dict=lstm_feed)

                print("Current training loss: %f" % loss)
                sys.stdout.flush()

                epoch_loss += loss
                train_batcher._step += 1

            # join threads
            sv.coord.request_stop()
            sv.coord.join(threads)
            sess.close()

def main(argv):
    print('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
    sys.stdout.flush()
    train()

if __name__ == '__main__':
    tf.app.run()

# srun python train.py --train_dir $HOME/data/pruned_pmc/train --dev_dir $HOME/data/pruned_pmc/dev --embeddings $HOME/data/embeddings/PMC-w2v.txt--train_eval
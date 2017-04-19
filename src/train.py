from __future__ import division
from __future__ import print_function
import sys
import time
import tensorflow as tf
import numpy as np
from models.batch_utils import SeqBatcher, Batcher
from models.bilstm import BiLSTM
from models.lstm import LSTM
from models.bilstm_char import BiLSTMChar
from models.cnn_char import CNNChar
from train_utils import *
import evaluate as evaluation
import json
import tf_utils

# FLAGS
# data directories
tf.app.flags.DEFINE_string('train_dir', '', 'directory containing preprocessed training data')
tf.app.flags.DEFINE_string('dev_dir', '', 'directory containing preprocessed dev data')
tf.app.flags.DEFINE_string('test_dir', '', 'directory containing preprocessed test data')

# directories for loading pre-trained models/checkpoints
tf.app.flags.DEFINE_string('model_dir', '', 'save model to this dir (if empty do not save)')
tf.app.flags.DEFINE_string('load_dir', '', 'load model from this dir (if empty do not load)')

# embeddings and dimensions
tf.app.flags.DEFINE_string('embeddings', '', 'path to embeddings file')
tf.app.flags.DEFINE_integer('embed_dim', 200, 'dimensions of the words embeddings')

# character embeddings
tf.app.flags.DEFINE_integer('char_dim', 0, 'character embedding dimension') # set to 25?
tf.app.flags.DEFINE_integer('char_tok_dim', 0, 'character token embedding dimension')
tf.app.flags.DEFINE_string('char_model', 'lstm', 'character embedding model (lstm, cnn)')
tf.app.flags.DEFINE_integer('shape_dim', 5, 'shape embedding dimension')

# TODO: should we be embedding other features?

# lstm layer dimensions
tf.app.flags.DEFINE_integer('lstm_dim', 2048, 'lstm internal dimension')
tf.app.flags.DEFINE_integer('max_seq_len', 30, 'should be same as seq_len')

# training
tf.app.flags.DEFINE_string('model', 'bilstm', 'which model to use [lstm, bilstm]')
tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size')
tf.app.flags.DEFINE_boolean('train_eval', False, 'whether to report train accuracy')
tf.app.flags.DEFINE_boolean('memmap_train', False, 'whether to load all training examples into memory')
tf.app.flags.DEFINE_string('master', '', 'use for Supervisor')
tf.app.flags.DEFINE_integer('max_epochs', 100, 'train for this many epochs')
tf.app.flags.DEFINE_boolean('until_convergence', False, 'whether to run until convergence')

# just run eval?
tf.app.flags.DEFINE_boolean('evaluate_only', False, 'whether to only run evaluation')

# features to use
tf.app.flags.DEFINE_boolean('use_geometric_feats', False, 'whether to use the geometric features')
tf.app.flags.DEFINE_boolean('use_lexicons', False, 'whether to use lexicon matching features')

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

tf.app.flags.DEFINE_float('char_input_dropout', 1.0, 'dropout for character embeddings')

# penalties
tf.app.flags.DEFINE_float('regularize_drop_penalty', 0.0, 'penalty for dropout regularization')

FLAGS = tf.app.flags.FLAGS

def run_train():
    # load preprocessed token, label, shape, char maps
    labels_str_id_map, labels_id_str_map, vocab_str_id_map, vocab_id_str_map, \
    shape_str_id_map, shape_id_str_map, char_str_id_map, char_id_str_map = load_intmaps(FLAGS.train_dir)

    # create intmaps for label types and bio (used later for evaluation, calculating F1 scores, etc.)
    type_int_int_map, bilou_int_int_map, type_set, bilou_set = create_type_maps(labels_str_id_map)

    # load the embeddings
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
        # todo add in dropout?
        if FLAGS.char_dim > 0 and FLAGS.char_model == "lstm":
            print("creating and training character embeddings")
            char_embedding_model = BiLSTMChar(char_domain_size, FLAGS.char_dim, int(FLAGS.char_tok_dim / 2))
        # elif FLAGS.char_dim > 0 and FLAGS.char_model == "cnn":
        #     char_embedding_model = CNNChar(char_domain_size, FLAGS.char_dim, FLAGS.char_tok_dim, layers_map[0][1]['width'])
        else:
            char_embedding_model = None
        char_embeddings = char_embedding_model.outputs if char_embedding_model is not None else None

        # create BiLSTM model
        if FLAGS.model == 'bilstm':
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
        elif FLAGS.model == 'lstm':
            model = LSTM(
                num_classes=labels_size,
                vocab_size=vocab_size,
                shape_domain_size=shape_domain_size,
                char_domain_size=char_domain_size,
                char_size=FLAGS.char_dim,
                embedding_size=FLAGS.embed_dim,
                shape_size=FLAGS.shape_dim,
                nonlinearity=FLAGS.nonlinearity,
                viterbi=False,  # viterbi=FLAGS.viterbi,
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

        frontend_opt_vars = [optimizer.get_slot(s, n) for n in optimizer.get_slot_names() for s in model_vars if
                             optimizer.get_slot(s, n) is not None]

        model_vars += frontend_opt_vars

        # load pretrained model if one is provided
        if FLAGS.load_dir:
            reader = tf.train.NewCheckpointReader(FLAGS.load_dir + ".tf")
            saved_var_map = reader.get_variable_to_shape_map()
            intersect_vars = [k for k in tf.all_variables() if
                              k.name.split(':')[0] in saved_var_map and k.get_shape() == saved_var_map[
                                  k.name.split(':')[0]]]
            leftovers = [k for k in tf.all_variables() if
                         k.name.split(':')[0] not in saved_var_map or k.get_shape() != saved_var_map[
                             k.name.split(':')[0]]]
            print("WARNING: Loading pretrained frontend, but not loading: ", map(lambda v: v.name, leftovers))
            frontend_loader = tf.train.Saver(var_list=intersect_vars)

        else:
            frontend_loader = tf.train.Saver(var_list=model_vars)

        frontend_saver = tf.train.Saver(var_list=model_vars)

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

            if FLAGS.load_dir != '':
                print("Deserializing model: " + FLAGS.load_dir + ".tf")
                frontend_loader.restore(sess, FLAGS.load_dir + ".tf")

            # load batches
            print()
            dev_batches, train_batches, num_dev_examples, num_train_examples \
                = load_batches(sess, train_batcher, train_eval_batcher, dev_batcher)

            # just run the evaluation
            if FLAGS.evaluate_only:
                if FLAGS.train_eval:
                    evaluation.run_evaluation(train_batches, FLAGS.layers2 != '', "(train)")
                print()
                evaluation.run_evaluation(dev_batches, FLAGS.layers2 != '', "(test)")
            # train a model
            else:
                best_score = 0
                total_iterations = 0

                # always train the front-end unless load dir was passed
                if FLAGS.load_dir == '' or (FLAGS.load_dir != '' and FLAGS.layers2 == ''):
                    best_score, training_iteration, train_speed = train(sess, sv, model, char_embedding_model,
                                                                        train_batches, dev_batches, num_train_examples,
                                                                        num_dev_examples, train_batcher,
                                                                        labels_str_id_map, labels_id_str_map,
                                                                        train_op, frontend_saver, vocab_str_id_map)
                    total_iterations += training_iteration
                    if FLAGS.model_dir:
                        print("Deserializing model: " + FLAGS.model_dir + "-frontend.tf")
                        frontend_saver.restore(sess, FLAGS.model_dir + "-frontend.tf")

        sv.coord.request_stop()
        sv.coord.join(threads)
        sess.close()

    total_time = time.time() - training_start_time
    if FLAGS.evaluate_only:
        print("Testing time: %d seconds" % (total_time))
    else:
        print("Training time: %d minutes, %d iterations (%3.2f minutes/iteration)" % (
        total_time / 60, total_iterations, total_time / (60 * total_iterations)))
        print("Avg training speed: %f examples/second" % (train_speed))
        print("Best dev F1: %2.2f" % (best_score * 100))

def train(sess, sv, model, char_embedding_model, train_batches, dev_batches, num_train_examples, num_dev_examples,
          train_batcher, labels_str_id_map, labels_id_str_map, train_op, frontend_saver, vocab_str_id_map):

    log_every = int(max(100, num_train_examples / 5))

    # start the training loop
    print("Training on %d pages (%d examples)" % (num_train_examples, num_train_examples))
    sys.stdout.flush()
    start_time = time.time()
    train_batcher._step = 1.0
    converged = False
    # keep track of how many training examples we've seen
    examples = 0
    log_every_running = log_every
    epoch_loss = 0.0
    num_lower = 0
    training_iteration = 0
    speed_num = 0.0
    speed_denom = 0.0
    total_iterations = 0
    max_lower = 6
    min_iters = 20
    best_score = 0
    # todo we might want this to be false for finetuning or something?
    update_context = True
    update_frontend = True
    while not sv.should_stop() and training_iteration < FLAGS.max_epochs and not (FLAGS.until_convergence and converged):
        # if we've gone through the entire dataset, update the epoch count (epoch = iteration here)
        if examples >= num_train_examples:
            training_iteration += 1
            print("iteration %d" % training_iteration)
            sys.stdout.flush()

            if FLAGS.train_eval:
                # print(len(train_batches))
                # print(len(train_batches[0]))
                # (sess, model, char_embedding_model, eval_batches, extra_text="")
                evaluation.run_evaluation(sess, model, char_embedding_model, train_batches, labels_str_id_map,
                                          labels_id_str_map, "TRAIN (iteration %d)" % training_iteration)
                print()
            weighted_f1, accuracy = evaluation.run_evaluation(sess, model, char_embedding_model, dev_batches, labels_str_id_map,
                                      labels_id_str_map, "TEST (iteration %d)" % training_iteration)
            print()
            # f1_micro, precision = evaluation.run_evaluation(dev_batches, update_context,
            #                                      "TEST (iteration %d)" % training_iteration)
            print("Avg training speed: %f examples/second" % (speed_num / speed_denom))

            # todo keep track of running best / convergence heuristic once i implement the eval
            if weighted_f1 > best_score:
                best_score = weighted_f1
                num_lower = 0
                if FLAGS.model_dir != '':
                    if update_frontend:
                        save_path = frontend_saver.save(sess, FLAGS.model_dir + "-frontend.tf")
                        print("Serialized model: %s" % save_path)
            else:
                num_lower += 1
            # if we've done the minimum number of iterations, check to see if the best score has converged
            if num_lower > max_lower and training_iteration > min_iters:
                converged = True
            #
            # # see if we have a better precision and save the model if so
            # if precision > best_precision:
            #     best_precision = precision
            #     if FLAGS.model_dir != '':
            #         if update_frontend and not update_context:
            #             save_path = frontend_saver.save(sess, FLAGS.model_dir + "-frontend-prec.tf")
            #             print("Serialized model: %s" % save_path)
            #         elif update_context and not update_frontend:
            #             save_path = context_saver.save(sess, FLAGS.model_dir + "-context-prec.tf")
            #             print("Serialized model: %s" % save_path)
            #         else:
            #             save_path = saver.save(sess, FLAGS.model_dir + "-prec.tf")
            #             print("Serialized model: %s" % save_path)

            # update per-epoch variables
            log_every_running = log_every
            examples = 0
            epoch_loss = 0.0
            start_time = time.time()

        if examples > log_every_running:
            speed_denom += time.time() - start_time
            speed_num += examples
            evaluation.print_training_error(examples, start_time, epoch_loss, train_batcher._step)
            sys.stdout.flush()
            log_every_running += log_every

        # train iteration
        # if we're not through an epoch yet, do training as usual
        label_batch, token_batch, shape_batch, char_batch, seq_len_batch, tok_lengths_batch,\
            widths_batch, heights_batch, wh_ratios_batch, x_coords_batch, y_coords_batch,\
            page_ids_batch, line_ids_batch, zone_ids_batch, \
            place_scores_batch, department_scores_batch, university_scores_batch, person_scores_batch= \
            train_batcher.next_batch() if FLAGS.memmap_train else sess.run(train_batcher.next_batch_op)

        # apply word dropout
        # create word dropout mask
        word_probs = np.random.random(token_batch.shape)
        drop_indices = np.where((word_probs > FLAGS.word_dropout))
        token_batch[drop_indices[0], drop_indices[1]] = vocab_str_id_map["<OOV>"]

        # TODO apply dropout to the rest of the features as well - are 0 features going to be an issue?

        # check that shapes look correct
        # print("label_batch_shape: ", label_batch.shape)
        # print("token batch shape: ", token_batch.shape)
        # print("shape batch shape: ", shape_batch.shape)
        # print("char batch shape: ", char_batch.shape)
        # print("seq_len batch shape: ", seq_len_batch.shape)
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
        # sys.stdout.flush()

        # reshape the features to be 3d tensors with 3rd dim = 1 (batch size) x (seq_len) x (1)
        # print("Reshaping features....")
        widths_batch = np.expand_dims(widths_batch, axis=2)
        heights_batch = np.expand_dims(heights_batch, axis=2)
        wh_ratios_batch = np.expand_dims(wh_ratios_batch, axis=2)
        x_coords_batch = np.expand_dims(x_coords_batch, axis=2)
        y_coords_batch = np.expand_dims(y_coords_batch, axis=2)
        page_ids_batch = np.expand_dims(page_ids_batch, axis=2)
        line_ids_batch = np.expand_dims(line_ids_batch, axis=2)
        zone_ids_batch = np.expand_dims(zone_ids_batch, axis=2)
        place_scores_batch = np.expand_dims(place_scores_batch, axis=2)
        department_scores_batch = np.expand_dims(department_scores_batch, axis=2)
        university_scores_batch = np.expand_dims(university_scores_batch, axis=2)
        person_scores_batch = np.expand_dims(person_scores_batch, axis=2)

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

        char_embedding_feeds = {} if FLAGS.char_dim == 0 else {
            char_embedding_model.input_chars: padded_char_batch,
            char_embedding_model.batch_size: batch_size,
            char_embedding_model.max_seq_len: batch_seq_len,
            char_embedding_model.token_lengths: tok_lengths_batch,
            char_embedding_model.max_tok_len: max_char_len,
            char_embedding_model.input_dropout_keep_prob: FLAGS.char_input_dropout
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

        lexicon_feats_feeds = {
            model.place_scores: place_scores_batch,
            model.department_scores: department_scores_batch,
            model.university_scores: university_scores_batch,
            model.person_scores: person_scores_batch
        }

        lstm_feed.update(char_embedding_feeds)

        if FLAGS.use_geometric_feats:
            lstm_feed.update(geometric_feats_feeds)

        if FLAGS.use_lexicons:
            lstm_feed.update(lexicon_feats_feeds)

        # print("Running training op:")
        sys.stdout.flush()
        # tf.Print(model.flat_sequence_lengths, [model.flat_sequence_lengths])
        _, loss = sess.run([train_op, model.loss], feed_dict=lstm_feed)

        epoch_loss += loss
        train_batcher._step += 1

    return best_score, training_iteration, speed_num/speed_denom

def main(argv):
    print('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
    sys.stdout.flush()
    run_train()

if __name__ == '__main__':
    tf.app.run()

# srun python train.py --train_dir $HOME/data/pruned_pmc/train --dev_dir $HOME/data/pruned_pmc/dev
    # --embeddings $HOME/data/embeddings/PMC-w2v.txt --train_eval
from __future__ import division
from __future__ import print_function
import sys
import time
import tensorflow as tf
import numpy as np
from models.batch_utils import SeqBatcher
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
tf.app.flags.DEFINE_integer('batch_size', 10, 'batch size')
tf.app.flags.DEFINE_boolean('train_eval', False, 'whether to report train accuracy')
tf.app.flags.DEFINE_boolean('memmap_train', True, 'whether to load all training examples into memory')

# hyperparams
tf.app.flags.DEFINE_string('nonlinearity', 'relu', 'nonlinearity function to use (tanh, sigmoid, relu)')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('l2', 0.0, 'l2 penalty')
tf.app.flags.DEFINE_float('beta1', 0.9, 'beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'beta2')
tf.app.flags.DEFINE_float('epsilon', 1e-8, 'epsilon')

# dropouts
tf.app.flags.DEFINE_float('char_input_dropout', 1.0, 'dropout for character embeddings')

# saving and loading models
tf.app.flags.DEFINE_string('model_dir', '', 'save model to this dir (if empty do not save)')
tf.app.flags.DEFINE_string('load_dir', '', 'load model from this dir (if empty do not load)')

FLAGS = tf.app.flags.FLAGS

def load_intmaps():
    print("Loading vocabulary maps...")
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
    return labels_str_id_map, labels_id_str_map, vocab_str_id_map, vocab_id_str_map, shape_str_id_map, shape_id_str_map, char_str_id_map, char_id_str_map

def load_embeddings(vocab_str_id_map):
    print("Loading embeddings...")
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
    return embeddings

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


# todo feed in embeddings here, since it seems I can't load them?
def make_predictions(sess, model, char_embedding_model, eval_batches, extra_text=""):
    predictions = []
    for b, (eval_label_batch, eval_token_batch, eval_shape_batch, eval_char_batch, eval_seq_len_batch, eval_tok_len_batch,
    eval_mask_batch) in enumerate(eval_batches):
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

        basic_feeds = {
            model.input_x1: eval_token_batch,
            model.input_x2: eval_shape_batch,
            model.input_y: eval_label_batch,
            model.input_mask: eval_mask_batch,
            model.max_seq_len: batch_seq_len,
            model.batch_size: batch_size,
            model.sequence_lengths: eval_seq_len_batch
        }

        basic_feeds.update(char_embedding_feeds)
        total_feeds = basic_feeds.copy()

        preds, scores = sess.run([model.predictions, model.unflat_scores], feed_dict=total_feeds)
        predictions.append(preds)

    return predictions

def load_batches(sess, train_batcher, train_eval_batcher, dev_batcher, pad_width=0):

    dev_batches = []
    # load all the dev batches into memory
    done = False
    print("Loading dev batches...")
    while not done:
        try:
            dev_batch = sess.run(dev_batcher.next_batch_op)
            dev_label_batch, dev_token_batch, dev_shape_batch, dev_char_batch, dev_seq_len_batch, dev_tok_len_batch = dev_batch
            mask_batch = np.zeros(dev_token_batch.shape)
            for i, seq_lens in enumerate(dev_seq_len_batch):
                start = pad_width
                for seq_len in seq_lens:
                    mask_batch[i, start:start + seq_len] = 1
                    # + (2 if FLAGS.start_end else 1) * pad_width
                    start += seq_len
            dev_batches.append((dev_label_batch, dev_token_batch, dev_shape_batch, dev_char_batch, dev_seq_len_batch,
                                dev_tok_len_batch, mask_batch))
        except:
            done = True

    print("Dev batches loaded.")

    print("Loading train batches...")
    train_batches = []
    if FLAGS.train_eval:
        # load all the train batches into memory
        done = False
        while not done:
            try:
                train_batch = sess.run(train_eval_batcher.next_batch_op)
                train_label_batch, train_token_batch, train_shape_batch, train_char_batch, train_seq_len_batch, train_tok_len_batch = train_batch
                mask_batch = np.zeros(train_token_batch.shape)
                for i, seq_lens in enumerate(train_seq_len_batch):
                    start = pad_width
                    for seq_len in seq_lens:
                        mask_batch[i, start:start + seq_len] = 1
                        # + (2 if FLAGS.start_end else 1) * pad_width
                        start += seq_len
                train_batches.append((train_label_batch, train_token_batch, train_shape_batch, train_char_batch,
                                  train_seq_len_batch, train_tok_len_batch, mask_batch))
            except Exception as e:
                done = True
    if FLAGS.memmap_train:
        train_batcher.load_and_bucket_data(sess)
    print("Train batches loaded.")

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
        train_batcher = SeqBatcher(FLAGS.train_dir, FLAGS.batch_size)
        dev_batcher = (FLAGS.dev_dir, FLAGS.batch_size)
        train_eval_batch_size = FLAGS.batch_size  # num_train_examples
        # train_eval_batcher = NodeBatcher(train_dir, seq_len_with_pad, train_eval_batch_size, num_epochs=1)
        train_eval_batcher = SeqBatcher(FLAGS.train_dir, train_eval_batch_size, num_buckets=0, num_epochs=1)

        # create character embedding model and train char embeddings:
        # todo this is broken, fix it and add it in when I get the rest of the network running
        # print("creating and training character embeddings")
        # char_embedding_model = BiLSTMChar(char_domain_size, FLAGS.char_dim, int(FLAGS.char_tok_dim / 2)) \
        #     if FLAGS.char_dim > 0 and FLAGS.char_model == "lstm" else \
        #     (CNNChar(char_domain_size, FLAGS.char_dim, FLAGS.char_tok_dim, layers_map[0][1]['width'])
        #      if FLAGS.char_dim > 0 and FLAGS.char_model == "cnn" else None)
        # char_embeddings = char_embedding_model.outputs if char_embedding_model is not None else None
        char_embeddings = None

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
            embeddings=embeddings)

        # Define Training procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        global_step_context = tf.Variable(0, name='context_agg_global_step', trainable=False)
        global_step_all = tf.Variable(0, name='context_agg_all_global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=FLAGS.beta1, beta2=FLAGS.beta2,
                                           epsilon=FLAGS.epsilon, name="optimizer")

        model_vars = [v for v in tf.all_variables() if 'context_agg' not in v.name]

        print("model vars: %d" % len(model_vars))
        print(map(lambda v: v.name, model_vars))
        get_trainable_params()

        tf.initialize_all_variables()

        sv = tf.python.train.Supervisor(logdir=FLAGS.model_dir if FLAGS.model_dir != '' else None,
                                        global_step=global_step,
                                        saver=None,
                                        save_model_secs=0,
                                        save_summaries_secs=0
                                        )




def main(argv):
    print('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
    train()

if __name__ == '__main__':
    tf.app.run()
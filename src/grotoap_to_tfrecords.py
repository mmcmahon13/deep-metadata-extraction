import glob
import multiprocessing
import sys
import re
import os
import tensorflow as tf
import numpy as np
import partial

from pdf_objects import *
from parse_docs_sax import *

tf.app.flags.DEFINE_string('trueviz_in_files', '', 'pattern to match trueviz document files')
tf.app.flags.DEFINE_string('out_dir', '', 'export tf protos')
tf.app.flags.DEFINE_string('load_vocab', '', 'directory containing embedding vocab files to load')
# TODO: need code to go through all the grotoap files and figure out the max number of sentences, words, pages, etc. -_-
tf.app.flags.DEFINE_integer('max_len', 500, 'maximum sequence length (in this case, max num words on a page?)')
tf.app.flags.DEFINE_integer('min_count', 5, 'replace tokens occuring less than this many times with <UNK>')
tf.app.flags.DEFINE_integer('num_threads', 12, 'max number of threads to use for parallel processing')
tf.app.flags.DEFINE_boolean('padding', 0, '0: no padding, 1: 0 pad to the right of seq, 2: 0 pad to the left')
tf.app.flags.DEFINE_boolean('normalize_digits', False, 'map all digits to 0')
tf.app.flags.DEFINE_boolean('lowercase', False, 'whether to lowercase')

FLAGS = tf.app.flags.FLAGS

PAD_STR = "<PAD>"
OOV_STR = "<OOV>"

# Helpers for creating Example objects (as defined in Pat's code)
feature = tf.train.Feature
sequence_example = tf.train.SequenceExample

def features(d): return tf.train.Features(feature=d)
def int64_feature(v): return feature(int64_list=tf.train.Int64List(value=v))
def feature_list(l): return tf.train.FeatureList(feature=l)
def feature_lists(d): return tf.train.FeatureLists(feature_list=d)

queue = multiprocessing.Queue()
queue.put(0)
lock = multiprocessing.Lock()


label_int_str_map = {}
token_int_str_map = {}
char_int_str_map = {}

def make_example(writer, page, label_map, token_map, shape_map, char_map, update_vocab, update_chars):
    # given a page from a document, make an example out of it (e.g. serialize it into a sequence of labeled word features)
    oov_count = 0

    # get all the words on the page, as well as their containing line and zone ids
    word_tups = []
    for zone in page.zones:
        for line in zone.lines:
            for word in line.words:
                word_tups.append((word, page.id, zone.id, line.id))

    # ignore padding for now and just let the batcher handle variable length sequences?
    max_len_with_pad = len(word_tups)
    max_word_len = max(map(len, [word.text for (word,_,_,_) in word_tups]))

    tokens = np.zeros(max_len_with_pad, dtype=np.int64)
    shapes = np.zeros(max_len_with_pad, dtype=np.int64)
    chars = np.zeros(max_len_with_pad * max_word_len, dtype=np.int64)
    intmapped_labels = np.zeros(max_len_with_pad, dtype=np.int64)
    page_lens = []
    tok_lens = []

    last_label = "O"
    labels = []
    current_tag = ''

    for word, page_id, zone_id, line_id in word_tups:
        # tokenize word using GENIA tokenization scheme (used by embeddings)
        # normalize word text - convert weird unicode to ASCII
        # check if word in vocab or not
        # generate BIO labels
        # get bounding box info (and other geometrical features?
        # get shape
        pass


# def doc_to_examples(total_docs, in_out):
#     label_map = {}
#     token_map = {}
#     shape_map = {}
#     char_map = {}
#
#     update_vocab = True
#     update_chars = True
#
#     # add the padding string to the token map
#     token_map[PAD_STR] = len(token_map)
#     token_int_str_map[token_map[PAD_STR]] = PAD_STR
#     # add the padding string to the character map
#     char_map[PAD_STR] = len(char_map)
#     char_int_str_map[char_map[PAD_STR]] = PAD_STR
#     # add the padding string to the shape map
#     shape_map[PAD_STR] = len(shape_map)
#     # if we want to predict the padding string, add it to the label map as well
#     if FLAGS.predict_pad:
#         label_map[PAD_STR] = len(label_map)
#         label_int_str_map[label_map[PAD_STR]] = PAD_STR
#
#     # add out of vocab string to the token, character maps
#     token_map[OOV_STR] = len(token_map)
#     token_int_str_map[token_map[OOV_STR]] = OOV_STR
#     char_map[OOV_STR] = len(char_map)
#     char_int_str_map[char_map[OOV_STR]] = OOV_STR
#
#     # load vocab (from the embeddings file)
#     if FLAGS.load_vocab != '':
#         update_vocab = False
#         with open(FLAGS.load_vocab, 'r') as f:
#             for line in f.readlines():
#                 word = line.strip().split(" ")[0]
#                 # token_map[word] = int(idx)
#                 # word = line.strip().split("\t")[0]
#                 if word not in token_map:
#                     # print("adding word %s" % word)
#                     token_map[word] = len(token_map)
#                     token_int_str_map[token_map[word]] = word
#
#     try:
#         in_f, out_path = in_out
#         writer = tf.python_io.TFRecordWriter(out_path)
#         print('Converting %s to %s' % (in_f, out_path))
#         doc = parse_doc(in_f)
#         for page in doc.pages:
#             make_example(writer, page, label_map, token_map, shape_map, char_map, update_vocab, update_chars)
#             # todo: use queue to keep track of how many docs we've done?
#         writer.close()
#         print('\nDone processing %s.' % in_f)
#     except KeyboardInterrupt:
#         return 'KeyboardException'
#
# def grotoap_to_examples():
#     if not os.path.exists(FLAGS.out_dir):
#         os.makedirs(FLAGS.out_dir)
#
#     # in_files = [in_f for in_f in FLAGS.in_files.split(',') if in_f]
#     # out_paths = [FLAGS.out_dir + '/' + out_f for out_f in FLAGS.out_files.split(',') if out_f]
#     in_files = sorted(glob.glob(FLAGS.trueviz_in_files))
#     out_files = ['%s/%s.proto' % (FLAGS.out_dir, in_f.split('/')[-1]) for in_f in in_files]
#
#     total_docs = len(in_files)
#
#     print('Starting file process threads using %d threads' % FLAGS.num_threads)
#     pool = multiprocessing.Pool(FLAGS.num_threads)
#     try:
#         pool.map_async(partial(doc_to_examples, total_docs), zip(in_files, out_files)).get(999999)
#         pool.close()
#         pool.join()
#     except KeyboardInterrupt:
#         pool.terminate()
#
#     # export the string->int maps to file
#     for f_str, id_map in [('label', label_map), ('token', token_map), ('shape', shape_map), ('char', char_map)]:
#         with open(FLAGS.out_dir + '/' + f_str + '.txt', 'w') as f:
#             [f.write(s + '\t' + str(i) + '\n') for (s, i) in id_map.items()]
#
# def main(argv):
#     print('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
#     if FLAGS.out_dir == '':
#         print('Must supply out_dir')
#         sys.exit(1)
#     grotoap_to_examples()


if __name__ == '__main__':
    tf.app.run()
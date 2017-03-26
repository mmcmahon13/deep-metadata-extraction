#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
from functools import partial
import codecs

import tensorflow as tf

from src.models.batch_utils import *
from src.processing.parse_docs_sax import *

tf.app.flags.DEFINE_string('grotoap_dir', '', 'top level directory containing grotoap dataset')
tf.app.flags.DEFINE_string('out_dir', '', 'export tf protos')
tf.app.flags.DEFINE_string('load_vocab', '', 'directory containing embedding vocab files to load')
tf.app.flags.DEFINE_boolean('use_lexicons', False, 'use string lexicon features')
tf.app.flags.DEFINE_integer('num_threads', 12, 'max number of threads to use for parallel processing')
# tf.app.flags.DEFINE_boolean('padding', 0, '0: no padding, 1: 0 pad to the right of seq, 2: 0 pad to the left')
# tf.app.flags.DEFINE_boolean('normalize_digits', False, 'map all digits to 0')
# tf.app.flags.DEFINE_boolean('lowercase', False, 'whether to lowercase')
tf.app.flags.DEFINE_boolean('debug', False, 'print debugging output')

FLAGS = tf.app.flags.FLAGS

# PAD_STR = "<PAD>"
OOV_STR = "<OOV>"

# Helpers for creating Example objects (as defined in Pat's code)
# feature = tf.train.Feature
# sequence_example = tf.train.SequenceExample
#
#
# def features(d): return tf.train.Features(feature=d)
# def int64_feature(v): return feature(int64_list=tf.train.Int64List(value=v))
# def feature_list(l): return tf.train.FeatureList(feature=l)
# def feature_lists(d): return tf.train.FeatureLists(feature_list=d)

# Emma's helpers
# def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
# def _float64_feature(value): return tf.train.Feature(float64_list=tf.train.Float64List(value=value))


# TODO: do we need these? i think pat only used them to keep track of the processing progress
queue = multiprocessing.Queue()
queue.put(0)
lock = multiprocessing.Lock()


# TODO since this will be multithreaded, protect these with locks?
label_map = {}
token_map = {}
shape_map = {}
char_map = {}
# just inverses of the maps for printing and junk
label_int_str_map = {}
token_int_str_map = {}
char_int_str_map = {}
shape_int_str_map = {}

def generate_bio(label, last_label):
    pass

def match_lexicons(token):
    pass

def serialize_example(writer, intmapped_labels, tokens, shapes, chars, page_lens, tok_lens,
                      widths, heights, wh_ratios, x_coords, y_coords, pages, lines, zones):
    example = tf.train.SequenceExample()

    fl_labels = example.feature_lists.feature_list["labels"]
    for l in intmapped_labels:
        fl_labels.feature.add().int64_list.value.append(l)

    fl_tokens = example.feature_lists.feature_list["tokens"]
    for t in tokens:
        fl_tokens.feature.add().int64_list.value.append(t)

    fl_shapes = example.feature_lists.feature_list["shapes"]
    for s in shapes:
        fl_shapes.feature.add().int64_list.value.append(s)

    fl_chars = example.feature_lists.feature_list["chars"]
    for c in chars:
        fl_chars.feature.add().int64_list.value.append(c)

    fl_seq_len = example.feature_lists.feature_list["seq_len"]
    for seq_len in page_lens:
        fl_seq_len.feature.add().int64_list.value.append(seq_len)

    fl_tok_len = example.feature_lists.feature_list["tok_len"]
    for tok_len in tok_lens:
        fl_tok_len.feature.add().int64_list.value.append(tok_len)

    # geometric feats
    fl_width = example.feature_lists.feature_list["widths"]
    for width in widths:
        fl_width.feature.add().float_list.value.append(width)

    fl_height = example.feature_lists.feature_list["heights"]
    for height in heights:
        fl_height.feature.add().float_list.value.append(height)

    fl_wh_ratio = example.feature_lists.feature_list["wh_ratios"]
    for wh_ratio in wh_ratios:
        fl_wh_ratio.feature.add().float_list.value.append(wh_ratio)

    fl_x_coords = example.feature_lists.feature_list["x_coords"]
    for x_coord in x_coords:
        fl_x_coords.feature.add().int64_list.value.append(x_coord)

    fl_y_coords = example.feature_lists.feature_list["y_coords"]
    for y_coord in y_coords:
        fl_y_coords.feature.add().int64_list.value.append(y_coord)

    # page, zone, line feats
    fl_page_id = example.feature_lists.feature_list["page_ids"]
    for page_id in pages:
        fl_page_id.feature.add().int64_list.value.append(page_id)

    fl_line_id = example.feature_lists.feature_list["line_ids"]
    for line_id in lines:
        fl_line_id.feature.add().int64_list.value.append(line_id)

    fl_zone_id = example.feature_lists.feature_list["zone_ids"]
    for zone_id in zones:
        fl_zone_id.feature.add().int64_list.value.append(zone_id)

    writer.write(example.SerializeToString())


def make_example(writer, page, update_vocab, update_chars):
    '''
    given a page from a document, make an example out of it (e.g. serialize it into a sequence of labeled word features)
    :param writer:
        the TFRecord writer
    :param page:
        a document page to make an example from
        dict mapping integers to characters
    :param update_vocab:
        whether or not to update the vocabulary with new words we find
    :param update_chars:
        whether or not to update the char map with new chars we find
    :return:
    '''

    # count how many words we encounter that fall outside of our embeddings vocab
    oov_count = 0

    # get all the words on the page, as well as their containing line and zone ids
    word_tups = []
    for zone in page.zones:
        for line in zone.lines:
            for word in line.words:
                word_tups.append((word, page.id, zone.id, line.id))

    if FLAGS.debug:
        print("words in page: ", len(word_tups))

    # ignore padding for now and just let the batcher handle variable length sequences?
    max_len_with_pad = len(word_tups)
    sum_word_len = sum(map(len, [word.text for (word,_,_,_) in word_tups]))

    ## for each sequence, keep track of the following word features:
    # vector of token ids
    tokens = np.zeros(max_len_with_pad, dtype=np.int64)
    # vector of shape ids
    shapes = np.zeros(max_len_with_pad, dtype=np.int64)
    # vector of character ids
    chars = np.zeros(sum_word_len, dtype=np.int64)
    # vector of label ids
    intmapped_labels = np.zeros(max_len_with_pad, dtype=np.int64)
    # TODO: add something for location (like a bin of regions or something)
    # geometrical information:
    widths = np.zeros(max_len_with_pad, dtype=np.float64)
    heights = np.zeros(max_len_with_pad, dtype=np.float64)
    wh_ratios = np.zeros(max_len_with_pad, dtype=np.float64)
    x_coords = np.zeros(max_len_with_pad, dtype=np.float64)
    y_coords = np.zeros(max_len_with_pad, dtype=np.float64)
    # enclosing region information (keep track of what zone, page, or line each word is in)
    pages = np.zeros(max_len_with_pad, dtype=np.int64)
    lines = np.zeros(max_len_with_pad, dtype=np.int64)
    zones = np.zeros(max_len_with_pad, dtype=np.int64)

    page_lens = [len(word_tups)]
    tok_lens = []

    last_label = "O"
    labels = []
    current_tag = ''
    char_start = 0

    # iterate over the words in the sequence and extract features, update vocab maps
    for i, (word, page_id, zone_id, line_id) in enumerate(word_tups):
        # get the word and its label
        token = word.text
        label = word.label

        # if FLAGS.debug:
        #     print(token, label)

        # TODO: tokenize word using GENIA tokenization scheme (used by embeddings)
        # TODO: normalize word text - convert weird unicode to ASCII
        token_normalized = token

        # check if word in vocab or not
        if token_normalized not in token_map:
            oov_count += 1
            if FLAGS.debug:
                print(token_normalized + " not in vocab.")
            if update_vocab:
                print("adding to vocab")
                token_map[token_normalized] = len(token_map)
                token_int_str_map[token_map[token_normalized]] = token_normalized

        # TODO: generate BIO labels
        # label_bilou = generate_bio(label, last_label)
        label_bilou = label

        # get bounding box info (and other geometrical features?)
        top_left = word.top_left
        bottom_right = word.bottom_right
        height = word.height()
        width = word.width()
        if height != 0:
            wh_ratio = width/height
        else:
            wh_ratio = 0
        (x, y) = word.centerpoint()

        # get shape and add to shape map
        token_shape = word.shape()
        if token_shape not in shape_map:  # and update_vocab:
            shape_map[token_shape] = len(shape_map)
            shape_int_str_map[shape_map[token_shape]] = token_shape

        # add characters to map
        for char in token:
            if char not in char_map and update_chars:
                char_map[char] = len(char_map)
                char_int_str_map[char_map[char]] = char

        # keep track of the lengths of the tokens
        tok_lens.append(len(token))

        # update the feature vectors:
        tokens[i] = token_map.get(token_normalized, token_map[OOV_STR])
        shapes[i] = shape_map[token_shape]
        # update char features
        chars[char_start:char_start+tok_lens[-1]] = [char_map.get(char, char_map[OOV_STR]) for char in token_normalized]
        # print(chars)
        char_start += tok_lens[-1]
        labels.append(label_bilou)
        last_label = label_bilou
        # update geometric features
        widths[i] = width
        heights[i] = height
        wh_ratios[i] = wh_ratio
        pages[i] = page_id
        lines[i] = line_id
        zones[i] = zone_id
        x_coords[i] = x
        y_coords[i] = y

    # bin the x and y coordinates
    x_bins = np.linspace(0, x_coords.max(), num=4)
    x_coords = np.digitize(x_coords, x_bins)
    y_bins = np.linspace(0, y_coords.max(), num=4)
    y_coords = np.digitize(y_coords, y_bins)

    # TODO: why are we intmapping the labels here? is it because of earlier BIO processing?

    for label in labels:
        if label not in label_map:
            label_map[label] = len(label_map)
            label_int_str_map[label_map[label]] = label

    intmapped_labels[:] = map(lambda s: label_map[s], labels)

    if FLAGS.debug:
        print("labels ", map(lambda t: label_int_str_map[t], intmapped_labels))
        print("tokens ", map(lambda t: token_int_str_map[t], tokens))
        print("chars", map(lambda t: char_int_str_map[t], chars))

        print("shapes ", map(lambda t: shape_int_str_map[t], shapes))
        print("widths ", widths)
        print("heights ", heights)
        print("w/h ratios ", wh_ratios)
        print("x coordinate bins ", x_coords)
        print("y coordinate bins ", y_coords)

        print("pages ", pages)
        print("lines ", lines)
        print("zones ", zones)

    print("serializing page ", page_id)
    serialize_example(writer,
                      intmapped_labels,
                      tokens,
                      shapes,
                      chars,
                      page_lens,
                      tok_lens,
                      widths,
                      heights,
                      wh_ratios,
                      x_coords,
                      y_coords,
                      pages,
                      lines,
                      zones)

    return len(word_tups), oov_count, 1


def doc_to_examples(in_file, writer):
    # convert all pages in a doc to examples

    update_vocab = True
    update_chars = True

    # # add the padding string to the token map
    # token_map[PAD_STR] = len(token_map)
    # token_int_str_map[token_map[PAD_STR]] = PAD_STR
    # # add the padding string to the character map
    # char_map[PAD_STR] = len(char_map)
    # char_int_str_map[char_map[PAD_STR]] = PAD_STR
    # # add the padding string to the shape map
    # shape_map[PAD_STR] = len(shape_map)
    # # if we want to predict the padding string, add it to the label map as well
    # if FLAGS.predict_pad:
    #     label_map[PAD_STR] = len(label_map)
    #     label_int_str_map[label_map[PAD_STR]] = PAD_STR

    # add out of vocab string to the token, character maps
    token_map[OOV_STR] = len(token_map)
    token_int_str_map[token_map[OOV_STR]] = OOV_STR

    char_map[OOV_STR] = len(char_map)
    char_int_str_map[char_map[OOV_STR]] = OOV_STR

    # load vocab (from the embeddings file)
    if FLAGS.load_vocab != '':
        print("loading vocab...")
        update_vocab = False
        with open(FLAGS.load_vocab, 'r') as f:
            for line in f.readlines():
                word = line.strip().split(" ")[0]
                # token_map[word] = int(idx)
                # word = line.strip().split("\t")[0]
                if word not in token_map:
                    # print("adding word %s" % word)
                    token_map[word] = len(token_map)
                    token_int_str_map[token_map[word]] = word

    try:
        # in_f, out_path = in_out
        # writer = tf.python_io.TFRecordWriter(out_path)
        # print('Converting %s to %s' % (in_f, out_path))
        print('Converting %s ' % in_file)
        doc = parse_doc(in_file)
        # for page in doc.pages:
        #     make_example(writer, page, label_map, token_map, shape_map, char_map, update_vocab, update_chars)
        # just start by trying first page only
        num_words, oov_count, _ = make_example(writer, doc.pages[0], update_vocab, update_chars)
        # writer.close()
        print('\nDone processing %s.' % in_file)
    except KeyboardInterrupt:
        return 'KeyboardException'

def dir_to_examples(root_dir, dir_out):
    (dir_path, out_path) = dir_out
    print("Converting directory %s to TFRecord %s" % (dir_path, out_path))
    writer = tf.python_io.TFRecordWriter(out_path)
    for root, dirs, files in os.walk(root_dir + os.sep + dir_path):
        for file in files:
            if '.cxml' in file:
                # filepath = FLAGS.grotoap_dir + os.sep + subdir + file
                filepath = root + os.sep + file
                doc_to_examples(filepath, writer)
    print("Done with directory %s" % dir_path)
    writer.close()


def grotoap_to_examples():
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    # train_writer = tf.python_io.TFRecordWriter(FLAGS.out_dir + os.sep + 'grotoap.train.proto')
    # test_writer = tf.python_io.TFRecordWriter(FLAGS.out_dir + os.sep + 'grotoap.test.proto')
    # dev_writer = tf.python_io.TFRecordWriter(FLAGS.out_dir + os.sep + 'grotoap.dev.proto')

    # in_files = []
    in_dirs = []
    out_files = []
    # TODO: point this at the grotoap directories, have one out file for each directory?
    for root, subdirs, files in os.walk(FLAGS.grotoap_dir):
        for subdir in subdirs:
            in_dirs.append(subdir)
        # for file in files:
        #     if '.cxml' in file:
        #         # filepath = FLAGS.grotoap_dir + os.sep + subdir + file
        #         filepath = subdir + os.sep + file
        #         in_files.append(filepath)

    # print(in_files)
    # write examples to 10 TFRecord files
    # for i,f in enumerate(in_files):
    #     # TODO: this doesn't work, it'll just overwrite the files every time a new writer is created...
    #     # out_files.append('%s/%s.proto' % (FLAGS.out_dir, i % 10))
    #     if i < 10000:
    #         out_files.append(FLAGS.out_dir + os.sep + 'grotoap.train.proto')
    #     elif i < 11500:
    #         out_files.append(FLAGS.out_dir + os.sep + 'grotoap.test.proto')
    #     else:
    #         out_files.append(FLAGS.out_dir + os.sep + 'grotoap.dev.proto')

    # print(out_files)

    # in_files = sorted(glob.glob(FLAGS.trueviz_in_files))
    out_files = ['%s/%s.proto' % (FLAGS.out_dir, in_f.split('/')[-1]) for in_f in in_dirs]

    total_dirs = len(in_dirs)

    print('Starting file process threads using %d threads' % FLAGS.num_threads)
    pool = multiprocessing.Pool(FLAGS.num_threads)
    try:
        # pool.map_async(partial(doc_to_examples, total_docs), zip(in_files, out_files)).get(999999)
        pool.map_async(partial(dir_to_examples, FLAGS.grotoap_dir), zip(in_dirs, out_files)).get(999999)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()

    # train_writer.close()
    # test_writer.close()
    # dev_writer.close()

def export_maps():
    # export the string->int maps to file
    print(len(label_map))
    print(len(token_map))
    print(len(char_map))
    print(len(shape_map))

    print('exporting string->int maps')
    for f_str, id_map in [('label', label_map), ('token', token_map), ('shape', shape_map), ('char', char_map)]:
        with codecs.open(FLAGS.out_dir + '/' + f_str + '.txt', 'w', 'utf-8') as f:
        # with open(FLAGS.out_dir + '/' + f_str + '.txt', 'w') as f:
            [f.write(s + '\t' + str(i) + '\n') for (s, i) in id_map.items()]

def main(argv):
    print('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
    if FLAGS.out_dir == '':
        print('Must supply out_dir')
        sys.exit(1)
    # test_doc_path = '/iesl/canvas/mmcmahon/data/GROTOAP2/grotoap2/dataset/00/1559601.cxml'
    # doc_to_examples(1, (test_doc_path, FLAGS.out_dir + '/examples.proto'))
    grotoap_to_examples()
    export_maps()
    # filename_queue = tf.train.string_input_producer([FLAGS.out_dir + '/iesl/canvas/mmcmahon/data/examples.proto'],
    #                                                 num_epochs=None)
    # labels, tokens, shapes, chars, tok_len, widths, heights, wh_ratios, x_coords, y_coords, page_ids, line_ids, zone_ids = parse_one_example(filename_queue)

if __name__ == '__main__':
    tf.app.run()

# python grotoap_to_tfrecords.py --out_dir $DATA_DIR --load_vocab /iesl/canvas/mmcmahon/embeddings/PubMed-w2v.txt

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
from functools import partial
import codecs
import sys
import numpy as np
import tensorflow as tf

# from src.models.batch_utils import *
from parse_docs_sax import *

tf.app.flags.DEFINE_string('grotoap_dir', '', 'top level directory containing grotoap dataset')
tf.app.flags.DEFINE_string('out_dir', '', 'export tf protos')
tf.app.flags.DEFINE_string('load_vocab', '', 'directory containing embedding vocab files to load')
tf.app.flags.DEFINE_boolean('use_lexicons', False, 'use string lexicon features')
tf.app.flags.DEFINE_integer('num_threads', 1, 'max number of threads to use for parallel processing')
# tf.app.flags.DEFINE_boolean('padding', 0, '0: no padding, 1: 0 pad to the right of seq, 2: 0 pad to the left')
# tf.app.flags.DEFINE_boolean('normalize_digits', False, 'map all digits to 0')
# tf.app.flags.DEFINE_boolean('lowercase', False, 'whether to lowercase')
tf.app.flags.DEFINE_boolean('debug', False, 'print debugging output')
tf.app.flags.DEFINE_boolean('bilou', False, 'encode the word labels in BILOU format')
tf.app.flags.DEFINE_integer('seq_len', 30, 'maximum sequence length')
tf.app.flags.DEFINE_integer('x_bins', 4, 'number of bins to use for x coordinate features')
tf.app.flags.DEFINE_integer('y_bins', 4, 'number of bins to use for y coordinate features')

FLAGS = tf.app.flags.FLAGS

# PAD_STR = "<PAD>"
OOV_STR = "<OOV>"


embeddings_counts = {}

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


def process_sequence(writer, word_tups, update_vocab, update_chars, token_map, token_int_str_map, label_map,
                     label_int_str_map, char_map, char_int_str_map, shape_map, shape_int_str_map, max_x, max_y,
                     min_x, min_y):
    oov_count = 0
    # ignore padding for now and just let the batcher handle variable length sequences?
    max_len_with_pad = len(word_tups)
    sum_word_len = sum(map(len, [word.text for (word, _, _, _) in word_tups]))

    ## for each sequence, keep track of the following word features:
    # vector of token ids
    tokens = np.zeros(max_len_with_pad, dtype=np.int64)
    # vector of shape ids
    shapes = np.zeros(max_len_with_pad, dtype=np.int64)
    # vector of character ids
    chars = np.zeros(sum_word_len, dtype=np.int64)
    # vector of label ids
    intmapped_labels = np.zeros(max_len_with_pad, dtype=np.int64)
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

    # todo should this just be a fixed len feature instead?
    page_lens = [len(word_tups)]
    tok_lens = []

    labels = []
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
                # print("adding to vocab")
                token_map[token_normalized] = len(token_map)
                token_int_str_map[token_map[token_normalized]] = token_normalized

        # keep track of how often embeddings are used so we can form a final
        # if token_normalized not in embeddings_counts:
        #     oov_count += 1
        # else:
        #     embeddings_counts[token_normalized] += 1

        label_bilou = label

        # get bounding box info (and other geometrical features?)
        top_left = word.top_left
        bottom_right = word.bottom_right
        height = word.height()
        width = word.width()
        if height != 0:
            wh_ratio = width / height
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
        chars[char_start:char_start + tok_lens[-1]] = [char_map.get(char, char_map[OOV_STR]) for char in
                                                       token_normalized]
        # print(chars)
        char_start += tok_lens[-1]
        labels.append(label_bilou)
        last_label = label_bilou
        # update geometric features
        widths[i] = width
        heights[i] = height
        # todo should we also bin widths, heights, wh ratios, etc.?
        wh_ratios[i] = wh_ratio
        pages[i] = page_id
        lines[i] = line_id
        zones[i] = zone_id
        x_coords[i] = x
        y_coords[i] = y

    # bin the x and y coordinates (4 bins from 0 to max x, y on page)
    x_bins = np.linspace(min_x, max_x, num=FLAGS.x_bins)
    x_coords = np.digitize(x_coords, x_bins)
    y_bins = np.linspace(min_y, max_y, num=FLAGS.y_bins)
    y_coords = np.digitize(y_coords, y_bins)

    # move this intmapping elsewhere?
    for label in labels:
        if label not in label_map:
            print("label %s not in vocab. adding..." % label)
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

    print("serializing sequence ", page_id)
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
    return oov_count

def make_example(writer, page, update_vocab, update_chars,
                 label_map, token_map, shape_map, char_map, label_int_str_map, token_int_str_map, char_int_str_map, shape_int_str_map):
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

    # run through all words in a page once to get max/min values for features we want to bin
    max_x = 0
    max_y = 0
    min_x = 100000
    min_y = 100000
    for zone in page.zones:
        for line in zone.lines:
            for word in line.words:
                (x,y) = word.centerpoint()
                if x > max_x:
                    max_x = x
                elif x < min_x:
                    min_x = x
                if y > max_y:
                    max_y = y
                elif y < min_y:
                    min_y = y

    # get all the words on the page, as well as their containing line and zone ids
    word_tups = []
    total_words = 0
    for zone in page.zones:
        for line in zone.lines:
            for word in line.words:
                # create sequences containing some number of words (chop page up arbitrarily)?
                if len(word_tups) < FLAGS.seq_len:
                    word_tups.append((word, page.id, zone.id, line.id))
                else:
                    oov_count += process_sequence(writer, word_tups, update_vocab, update_chars, token_map, token_int_str_map, label_map,
                     label_int_str_map, char_map, char_int_str_map, shape_map, shape_int_str_map, max_x, max_y, min_x, min_y)
                    word_tups = [(word, page.id, zone.id, line.id)]
                total_words += 1

    if FLAGS.debug:
        print(len(label_map))
        print(len(token_map))
        print(len(char_map))
        print(len(shape_map))

    return total_words, oov_count, 1


def doc_to_examples(in_file, writer, label_map, token_map, shape_map, char_map, label_int_str_map,
                    token_int_str_map, char_int_str_map, shape_int_str_map):
    # convert all pages in a doc to examples

    update_vocab = False
    update_chars = True

    try:
        print('Converting %s ' % in_file)
        doc = parse_doc(in_file)

        # convert labels to BILOU
        if FLAGS.bilou:
            words_to_bilou(doc)

        # check for dictionary matches if we want to
        if FLAGS.use_lexicons:
            place_set, department_set, university_set, person_set = load_dictionaries()
            match_dictionaries(doc, place_set, department_set, university_set, person_set)

        # just start by trying first page only
        num_words, oov_count, _ = make_example(writer, doc.pages[0], update_vocab, update_chars,
                                               label_map, token_map, shape_map, char_map, label_int_str_map,
                                               token_int_str_map, char_int_str_map, shape_int_str_map)
        # writer.close()
        print('\nDone processing %s.' % in_file)
        print(num_words, oov_count)
        return num_words, oov_count
    except KeyboardInterrupt:
        return 'KeyboardException'

def dir_to_examples(root_dir, label_map, token_map, shape_map, char_map, label_int_str_map, token_int_str_map,
                    char_int_str_map, shape_int_str_map,
                    dir_out):
    '''
    Convert a directory of TrueViz document files to one TFRecord file containing examples from all of them

    :param root_dir: directory of files
    :param dir_out: target TFRecord file
    :return:
    '''
    (dir_path, out_path) = dir_out
    print("Converting directory %s to TFRecord %s" % (dir_path, out_path))
    writer = tf.python_io.TFRecordWriter(out_path)
    tot_words = 0
    tot_oov = 0
    for root, dirs, files in os.walk(root_dir + os.sep + dir_path):
        for file in files:
            if '.cxml' in file:
                # filepath = FLAGS.grotoap_dir + os.sep + subdir + file
                filepath = root + os.sep + file
                num_words, oov_count = doc_to_examples(filepath, writer, label_map, token_map, shape_map, char_map,
                                                       label_int_str_map, token_int_str_map, char_int_str_map, shape_int_str_map)
                tot_words += num_words
                tot_oov += oov_count
    print("Done with directory %s" % dir_path)
    coverage = 1 - tot_oov/tot_words
    print("Embeddings coverage: %f" % coverage)
    writer.close()


def grotoap_to_examples(label_map, token_map, shape_map, char_map, label_int_str_map, token_int_str_map,
                        char_int_str_map, shape_int_str_map, use_threads=False):
    '''
    Convert entire GROTOAP set to TFRecords (iterate over the subdirectors 00, 01, ..., 99 and create a TFRecord for each)
    :return:
    '''
    # load vocab (from the embeddings file)
    # todo keep track of usage counts, remove unused embeddings from the saved vocab?
    if FLAGS.load_vocab != '':
        print("loading vocab...")
        update_vocab = False
        with open(FLAGS.load_vocab, 'r') as f:
            for line in f.readlines():
                word = line.strip().split(" ")[0]
                if word not in token_map:
                    # print("adding word %s" % word)
                    token_map[word] = len(token_map)
                    token_int_str_map[token_map[word]] = word

     # add out of vocab string to the token, character maps
    token_map[OOV_STR] = len(token_map)
    token_int_str_map[token_map[OOV_STR]] = OOV_STR

    char_map[OOV_STR] = len(char_map)
    char_int_str_map[char_map[OOV_STR]] = OOV_STR

    print("%d words in vocab" % len(token_map))

    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    in_dirs = []

    # should point this at the directory containing the numbered subdirectories
    for root, subdirs, files in os.walk(FLAGS.grotoap_dir):
        for subdir in subdirs:
            in_dirs.append(subdir)

    out_files = ['%s/%s.proto' % (FLAGS.out_dir, in_f.split('/')[-1]) for in_f in in_dirs]

    total_dirs = len(in_dirs)

    # TODO threading doesn't save the maps correctly
    # TODO I think if we want to use multithreading we need to iterate over the data once to create the maps
    if use_threads:
        print('Starting file process threads using %d threads' % FLAGS.num_threads)
        pool = multiprocessing.Pool(FLAGS.num_threads)
        try:
            # pool.map_async(partial(doc_to_examples, total_docs), zip(in_files, out_files)).get(999999)
            pool.map_async(partial(dir_to_examples, FLAGS.grotoap_dir, label_map, token_map, shape_map, char_map, label_int_str_map, token_int_str_map,
                    char_int_str_map, shape_int_str_map), zip(in_dirs, out_files)).get(999999)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
    else:
        for (in_dir, out_file) in zip(in_dirs, out_files):
            dir_to_examples(FLAGS.grotoap_dir, label_map, token_map, shape_map, char_map, label_int_str_map, token_int_str_map,
                    char_int_str_map, shape_int_str_map, (in_dir, out_file))

def export_maps(label_map, token_map, shape_map, char_map):
    # export the string->int maps to file
    print(len(label_map))
    print(len(token_map))
    print(len(char_map))
    print(len(shape_map))

    print('exporting string->int maps')
    for f_str, id_map in [('label', label_map), ('token', token_map), ('shape', shape_map), ('char', char_map)]:
        print(f_str)
        with codecs.open(FLAGS.out_dir + '/' + f_str + '.txt', 'w', 'utf-8') as f:
            [f.write(s + '\t' + str(i) + '\n') for (s, i) in id_map.items()]

def main(argv):
    print('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
    if FLAGS.out_dir == '':
        print('Must supply out_dir')
        sys.exit(1)
    # test_doc_path = '/iesl/canvas/mmcmahon/data/GROTOAP2/grotoap2/dataset/00/1559601.cxml'
    # doc_to_examples(1, (test_doc_path, FLAGS.out_dir + '/examples.proto'))

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

    if FLAGS.num_threads > 1:
        use_threads = True
    else:
        use_threads = False

    grotoap_to_examples(label_map, token_map, shape_map, char_map, label_int_str_map, token_int_str_map, char_int_str_map, shape_int_str_map, use_threads)
    export_maps(label_map, token_map, shape_map, char_map)
    # filename_queue = tf.train.string_input_producer([FLAGS.out_dir + '/iesl/canvas/mmcmahon/data/examples.proto'],
    #                                                 num_epochs=None)
    # labels, tokens, shapes, chars, tok_len, widths, heights, wh_ratios, x_coords, y_coords, page_ids, line_ids, zone_ids = parse_one_example(filename_queue)

if __name__ == '__main__':
    tf.app.run()

# python grotoap_to_tfrecords.py --out_dir $DATA_DIR/pruned_pmc/train --load_vocab $DATA_DIR/pruned_PMC.txt --grotoap_dir /iesl/canvas/mmcmahon/data/GROTOAP2/grotoap2/dataset --bilou
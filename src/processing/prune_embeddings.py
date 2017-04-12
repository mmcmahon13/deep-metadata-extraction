import codecs
import tensorflow as tf
import os
from parse_docs_sax import *

tf.app.flags.DEFINE_string('grotoap_dir', '', 'top level directory containing grotoap dataset')
tf.app.flags.DEFINE_string('out_dir', '', 'directory for output file')
tf.app.flags.DEFINE_string('out_file', '', 'export final vocab to this file')
tf.app.flags.DEFINE_string('embeddings', '', 'directory containing embedding vocab files to load')
tf.app.flags.DEFINE_integer('min_count', 1, 'minimum occurence count to be included in vocab')

FLAGS = tf.app.flags.FLAGS

embeddings_counts = {}
token_map = {}


def count_vocab(root_dir, dir_path):
    # run through embeddings and count their occurrences in the data so we can eliminate useless entries
    for root, dirs, files in os.walk(root_dir + os.sep + dir_path):
        for file in files:
            if '.cxml' in file:
                # filepath = FLAGS.grotoap_dir + os.sep + subdir + file
                filepath = root + os.sep + file
                doc = parse_doc(filepath)
                print(len(doc.words()))
                for word in doc.words():
                    token = word.text
                    if token in embeddings_counts:
                        embeddings_counts[token] += 1


def main(argv):
    print('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
    in_dirs = []

    # should point this at the directory containing the numbered subdirectories
    for root, subdirs, files in os.walk(FLAGS.grotoap_dir):
        for subdir in subdirs:
            in_dirs.append(subdir)

    print("num directories: %d" % len(in_dirs))

    # read in embeddings
    with open(FLAGS.embeddings, 'r') as f:
        print("loading vocab...")
        for line in f.readlines():
            word = line.strip().split(" ")[0]
            if word not in embeddings_counts:
                embeddings_counts[word] = 0

    print("counting embeddings...")
    # count how many times each embedding is used
    for dir in in_dirs:
        print(dir)
        count_vocab(FLAGS.grotoap_dir, dir)

    # build final vocab from the most-used embeddings
    for word in embeddings_counts:
        if embeddings_counts[word] >= FLAGS.min_count:
            token_map[word] = len(token_map)

    print("final vocab size: %d" % len(token_map))

    print("exporting vocab...")
    with codecs.open(FLAGS.out_dir + '/' + FLAGS.out_file, 'w', 'utf-8') as f:
        for s in token_map:
            f.write(s + '\n')
    print("done.")

if __name__ == '__main__':
    tf.app.run()


# python prune_embeddings.py --embeddings /iesl/canvas/mmcmahon/embeddings/PMC-w2v.txt --grotoap_dir $DATA_DIR/grotoap_test/grotoap2/dataset --out_dir $DATA_DIR --out_file 'pruned_PMC.txt'
import argparse
import os

import pickle

from pdf_objects import *
from parse_docs_sax import *

def create_vocab(embeddings_filepath):
    word_count = 0
    word_int_map = {}

    print("Creating vocab for %s" % embeddings_filepath)
    with open(embeddings_filepath, 'r') as ef:
        for line in ef:
            comps = line.split(" ")
            word = comps[0]
            if not word in word_int_map:
                word_count += 1
                word_int_map[word] = word_count
    return word_int_map

def check_coverage(trueviz_dir_path, vocab_intmap):
    num_words = 0
    num_in_vocab = 0
    for root, subdirs, files in os.walk(trueviz_dir_path):
        for tv_file in files:
            if '.cxml' in tv_file:
                src_file_path = root + os.sep + tv_file
                doc = parse_doc(src_file_path)
                for word in doc.words():
                    if word.text in vocab_intmap:
                        num_in_vocab += 1
                    num_words += 1
    print("Embeddings coverage: %f" % num_in_vocab/num_words)


def main():
    arg_parser = argparse.ArgumentParser(description='Get the vocab from an embeddings file')
    arg_parser.add_argument('embeddings_file_path')
    arg_parser.add_argument('target_path')
    arg_parser.add_argument('--dir', type=str, help='true-viz directory (if you want to check embeddings coverage)', required=False)
    args = arg_parser.parse_args()

    vocab_intmap = create_vocab(args.embeddings_file_path)
    pickle.dumps(vocab_intmap, open(args.target_path, 'wb'))
    check_coverage(vocab_intmap, args.dir)


if __name__ == "__main__":
    main()
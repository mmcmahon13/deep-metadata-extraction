import numpy as np
import tensorflow as tf
from collections import defaultdict
from random import shuffle
import sys

FLAGS = tf.app.flags.FLAGS

# just for testing that things are being serialized as expected
# def parse_one_example(filename_queue):
#     reader = tf.TFRecordReader()
#     key, record_string = reader.read(filename_queue)
#     features = {
#         'labels': tf.FixedLenSequenceFeature([], tf.int64),
#         'tokens': tf.FixedLenSequenceFeature([], tf.int64),
#         'shapes': tf.FixedLenSequenceFeature([], tf.int64),
#         'chars': tf.FixedLenSequenceFeature([], tf.int64),
#         # 'seq_len': tf.FixedLenSequenceFeature([], tf.int64),
#         'tok_len': tf.FixedLenSequenceFeature([], tf.int64),
#         'widths': tf.FixedLenSequenceFeature([], tf.float32),
#         'heights': tf.FixedLenSequenceFeature([], tf.float32),
#         'wh_ratios': tf.FixedLenSequenceFeature([], tf.float32),
#         'page_ids': tf.FixedLenSequenceFeature([], tf.int64),
#         'line_ids': tf.FixedLenSequenceFeature([], tf.int64),
#         'zone_ids': tf.FixedLenSequenceFeature([], tf.int64)
#     }
#
#     _, example = tf.parse_single_sequence_example(serialized=record_string, sequence_features=features)
#     labels = example['labels']
#     tokens = example['tokens']
#     shapes = example['shapes']
#     chars = example['chars']
#     # seq_len = example['seq_len']
#     tok_len = example['tok_len']
#     widths = example['widths']
#     heights = example['heights']
#     wh_ratios = example['wh_ratios']
#     page_ids = example['page_ids']
#     line_ids = example['line_ids']
#     zone_ids = example['zone_ids']
#
#     # context = c['context']
#     return labels, tokens, shapes, chars, tok_len, widths, heights, wh_ratios, page_ids, line_ids, zone_ids
    # return labels, tokens, labels, labels, labels

# Emma Strubell's implementation of a sequence batcher with shuffling (since it doesn't seem to exist in TF)
class SeqBatcher(object):
    def __init__(self, in_dir, batch_size, num_buckets=0, num_epochs=None):
        self._batch_size = batch_size
        self.num_buckets = num_buckets
        self._epoch = 0
        self._step = 1.
        self.num_epochs = num_epochs
        in_file = [in_dir + '/examples.proto']
        self.next_batch_op = self.input_pipeline(in_file, self._batch_size, self.num_buckets, self.num_epochs)

    def example_parser(self, filename_queue):
        reader = tf.TFRecordReader()
        key, record_string = reader.read(filename_queue)
        features = {
            'labels': tf.FixedLenSequenceFeature([], tf.int64),
            'tokens': tf.FixedLenSequenceFeature([], tf.int64),
            'shapes': tf.FixedLenSequenceFeature([], tf.int64),
            'chars': tf.FixedLenSequenceFeature([], tf.int64),
            'seq_len': tf.FixedLenSequenceFeature([], tf.int64),
            'tok_len': tf.FixedLenSequenceFeature([], tf.int64),
            'widths': tf.FixedLenSequenceFeature([], tf.float32),
            'heights': tf.FixedLenSequenceFeature([], tf.float32),
            'wh_ratios': tf.FixedLenSequenceFeature([], tf.float32),
            'x_coords': tf.FixedLenSequenceFeature([], tf.int64),
            'y_coords': tf.FixedLenSequenceFeature([], tf.int64),
            'page_ids': tf.FixedLenSequenceFeature([], tf.int64),
            'line_ids': tf.FixedLenSequenceFeature([], tf.int64),
            'zone_ids': tf.FixedLenSequenceFeature([], tf.int64)
        }

        _, example = tf.parse_single_sequence_example(serialized=record_string, sequence_features=features)
        labels = example['labels']
        tokens = example['tokens']
        shapes = example['shapes']
        chars = example['chars']
        seq_len = example['seq_len']
        tok_len = example['tok_len']
        widths = example['widths']
        heights = example['heights']
        wh_ratios = example['wh_ratios']
        x_coords = example['x_coords']
        y_coords = example['y_coords']
        page_ids = example['page_ids']
        line_ids = example['line_ids']
        zone_ids = example['zone_ids']

        # context = c['context']
        return labels, tokens, shapes, chars, seq_len, tok_len, widths, heights, wh_ratios, x_coords, y_coords, page_ids, line_ids, zone_ids
        # return labels, tokens, labels, labels, labels

    def input_pipeline(self, filenames, batch_size, num_buckets, num_epochs=None):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        labels, tokens, shapes, chars, seq_len, tok_len, widths, heights, wh_ratios, x_coords, y_coords, page_ids, \
        line_ids, zone_ids = self.example_parser(filename_queue)
        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 12 * batch_size

        # next_batch = tf.train.batch([labels, tokens, shapes, chars, seq_len], batch_size=batch_size, capacity=capacity,
        #                                 dynamic_pad=True, allow_smaller_final_batch=True)

        if num_buckets == 0:
            next_batch = tf.train.batch([labels, tokens, shapes, chars, seq_len, tok_len], batch_size=batch_size, capacity=capacity,
                                        dynamic_pad=True, allow_smaller_final_batch=True)
        else:
            bucket, next_batch = tf.contrib.training.bucket([labels, tokens, shapes, chars, seq_len, tok_len, widths, heights,
                                                             wh_ratios, x_coords, y_coords, page_ids, line_ids, zone_ids],
                                                            np.random.randint(num_buckets),
                                                        batch_size, num_buckets, num_threads=1, capacity=capacity,
                                                        dynamic_pad=True, allow_smaller_final_batch=False)
        return next_batch
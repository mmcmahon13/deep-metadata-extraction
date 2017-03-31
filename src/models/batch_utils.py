import numpy as np
import tensorflow as tf
from collections import defaultdict
from random import shuffle
import os
import sys

FLAGS = tf.app.flags.FLAGS

# Emma Strubell's batcher
class Batcher(object):
    def __init__(self, in_dir, batch_size, num_epochs=None):
        self._batch_size = batch_size
        self._epoch = 0
        self._step = 0.
        self._data = defaultdict(list)
        self._starts = {}
        self._ends = {}
        self._bucket_probs = {}
        self.sequence_batcher = SeqBatcher(in_dir, batch_size, 0, num_epochs=1)

    def load_and_bucket_data(self, sess):
        done = False
        i = 0
        while not done:
            try:
                batch = sess.run(self.sequence_batcher.next_batch_op)
                self._data[batch[0].shape[1]].append(batch)
                i += 1
            except Exception as e:
                done = True
        # now flatten
        for seq_len, batches in self._data.items():
            self._data[seq_len] = [(label_batch[i], token_batch[i], shape_batch[i], char_batch[i], seq_len_batch[i], tok_len_batch[i],
                                    width_batch[i], height_batch[i], wh_ratio_batch[i], x_coord_batch[i], y_coord_batch[i],
                                    page_id_batch[i], line_id_batch[i], zone_id_batch[i])
                                  for (label_batch, token_batch, shape_batch, char_batch, seq_len_batch, tok_len_batch,
                                       width_batch, height_batch, wh_ratio_batch, x_coord_batch, y_coord_batch,
                                       page_id_batch, line_id_batch, zone_id_batch) in batches
                                  for i in range(label_batch.shape[0])]

        self.reset_batch_pointer()

    def next_batch(self):
        if sum(self._bucket_probs.values()) == 0:
            self.reset_batch_pointer()
        # select bucket to create batch from
        self._step += 1
        bucket = self.select_bucket()
        batch = self._data[bucket][self._starts[bucket]:self._ends[bucket]]
        # update pointers
        self._starts[bucket] = self._ends[bucket]
        self._ends[bucket] = min(self._ends[bucket] + self._batch_size, len(self._data[bucket]))
        self._bucket_probs[bucket] = max(0, self._ends[bucket] - self._starts[bucket])

        # labels, tokens, shapes, chars, seq_len, tok_len, widths, heights, wh_ratios, x_coords, y_coords, page_ids, line_ids, zone_ids
        _label_batch = np.array([b[0] for b in batch])
        _token_batch = np.array([b[1] for b in batch])
        _shape_batch = np.array([b[2] for b in batch])
        _char_batch = np.array([b[3] for b in batch])
        _seq_len_batch = np.array([b[4] for b in batch])
        _tok_len_batch = np.array([b[5] for b in batch])
        _width_batch = np.array([b[6] for b in batch])
        _height_batch = np.array([b[7] for b in batch])
        _wh_ratio_batch = np.array([b[8] for b in batch])
        _x_coord_batch = np.array([b[9] for b in batch])
        _y_coord_batch = np.array([b[10] for b in batch])
        _page_id_batch = np.array([b[11] for b in batch])
        _line_id_batch = np.array([b[12] for b in batch])
        _zone_id_batch = np.array([b[13] for b in batch])

        batch = (_label_batch, _token_batch, _shape_batch, _char_batch, _seq_len_batch, _tok_len_batch,
                 _width_batch, _height_batch, _wh_ratio_batch, _x_coord_batch, _y_coord_batch, _page_id_batch,
                 _line_id_batch, _zone_id_batch)

        return batch

    def reset_batch_pointer(self):
        # shuffle each bucket
        for bucket in self._data.values():
            shuffle(bucket)
        self._epoch += 1
        self._step = 0.
        # print('\nStarting epoch ' + str(self._epoch))
        self._starts = {i: 0 for i in self._data.keys()}
        self._ends = {i: min(self._batch_size, len(examples)) for i, examples in self._data.items()}
        self._bucket_probs = {i: len(l) for (i, l) in self._data.items()}

    def select_bucket(self):
        buckets, weights = zip(*[(i, p) for i, p in self._bucket_probs.items() if p > 0])
        total = float(sum(weights))
        probs = [w / total for w in weights]
        bucket = np.random.choice(buckets, p=probs)
        return bucket

# Emma Strubell's implementation of a sequence batcher with shuffling (since it doesn't seem to exist in TF)
class SeqBatcher(object):
    def __init__(self, in_dir, batch_size, num_buckets=0, num_epochs=None):
        self._batch_size = batch_size
        self.num_buckets = num_buckets
        self._epoch = 0
        self._step = 1.
        self.num_epochs = num_epochs
        # todo change this to take multiple proto files from the same directory
        in_files = []
        for root, dirs, files in os.walk(in_dir):
            for file in files:
                if file.contains('.proto'):
                    in_files.append(file)
        # in_file = [in_dir + '/examples.proto']
        self.next_batch_op = self.input_pipeline(in_files, self._batch_size, self.num_buckets, self.num_epochs)

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
            next_batch = tf.train.batch([labels, tokens, shapes, chars, seq_len, tok_len, widths, heights,
                                        wh_ratios, x_coords, y_coords, page_ids, line_ids, zone_ids],
                                        batch_size=batch_size, capacity=capacity,
                                        dynamic_pad=True, allow_smaller_final_batch=True)
        else:
            bucket, next_batch = tf.contrib.training.bucket([labels, tokens, shapes, chars, seq_len, tok_len, widths, heights,
                                                             wh_ratios, x_coords, y_coords, page_ids, line_ids, zone_ids],
                                                            np.random.randint(num_buckets),
                                                        batch_size, num_buckets, num_threads=1, capacity=capacity,
                                                        dynamic_pad=True, allow_smaller_final_batch=False)
        return next_batch
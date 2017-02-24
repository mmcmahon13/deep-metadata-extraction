import tensorflow as tf
import numpy as np

# a CNN based on Kim Yoon’s Convolutional Neural Networks for Sentence Classification
# adapted from Denny Britz's tutorial
# http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    # def __init__(
    #         self, sequence_length, num_classes, vocab_size,
    #         embedding_size, filter_sizes, num_filters):

    # Implementation...
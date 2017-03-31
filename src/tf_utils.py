from __future__ import division
import tensorflow as tf
import numpy as np

eps = 1e-5

'''
Emma Strubell's utitlity functions (needed for her models)
'''

def apply_nonlinearity(parameters, nonlinearity_type):
    if nonlinearity_type == "relu":
        return tf.nn.relu(parameters, name="relu")
    elif nonlinearity_type == "tanh":
        return tf.nn.tanh(parameters, name="tanh")
    elif nonlinearity_type == "sigmoid":
        return tf.nn.sigmoid(parameters, name="sigmoid")


def embedding_values(shape, old=False):
    if old:
        embeddings = np.multiply(np.add(np.random.rand(shape[0], shape[1]).astype('float32'), -0.1), 0.01)
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        embeddings = drange * np.random.uniform(low=-1.0, high=1.0, size=shape).astype('float32')
    return embeddings

# TODO this fails with files >2GB
# shard the embeddings?
def initialize_embeddings(shape, name, pretrained=None, old=False):
    zero_pad = tf.constant(0.0, dtype=tf.float32, shape=[1, shape[1]])
    if pretrained is None:
        embeddings = embedding_values(shape, old)
    else:
        embeddings = pretrained
    return tf.concat(0, [zero_pad, tf.get_variable(name=name, initializer=embeddings)])


def initialize_weights(shape, name, init_type, gain="1.0", divisor=1.0):
    if init_type == "random":
        return tf.get_variable(name, initializer=tf.truncated_normal(shape, stddev=0.1))
    if init_type == "xavier":
        # shape_is_tensor = issubclass(type(shape), tf.Tensor)
        # rank = len(shape.get_shape()) if shape_is_tensor else len(shape)
        # if rank == 4:
        #     return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    if init_type == "identity":
        middle = int(shape[1] / 2)
        if shape[2] == shape[3]:
            array = np.zeros(shape, dtype='float32')
            identity = np.eye(shape[2], shape[3])
            array[0, middle] = identity
        else:
            m1 = divisor / shape[2]
            m2 = divisor / shape[3]
            sigma = eps*m2
            array = np.random.normal(loc=0, scale=sigma, size=shape).astype('float32')
            for i in range(shape[2]):
                for j in range(shape[3]):
                    if int(i*m1) == int(j*m2):
                        array[0, middle, i, j] = m2
        return tf.get_variable(name, initializer=array)
    if init_type == "varscale":
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    if init_type == "orthogonal":
        gain = np.sqrt(2) if gain == "relu" else 1.0
        array = np.zeros(shape, dtype='float32')
        random = np.random.normal(0.0, 1.0, (shape[2], shape[3])).astype('float32')
        u, _, v_t = np.linalg.svd(random, full_matrices=False)
        middle = int(shape[1] / 2)
        array[0, middle] = gain * v_t
        return tf.get_variable(name, initializer=array)
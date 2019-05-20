"""
Created on 2019-04-29
@author: hty
"""
import tensorflow as tf
 
def weight_variable(shape, name):
    weight = tf.get_variable(name, shape, initializer=tf.keras.initializers.he_normal(), dtype=tf.float32)
    return weight
    
    
def bias_variable(shape, name):
    bias = tf.get_variable(name, shape, initializer=tf.zeros_initializer(), dtype=tf.float32)
    return bias
    
    
def relu(x):
    return tf.nn.relu(x)
    
    
def lrelu(x, alpha=0.05):
    return tf.nn.leaky_relu(x, alpha)


def prelu(x, a_name):
    alphas = tf.get_variable(a_name, x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = relu(x)
    neg = alphas * (x - abs(x)) * 0.5
    return pos + neg


def conv2d(x, shape, name, stride=[1, 1, 1, 1], pad='SAME', activition='lrelu', alpha=0.05, use_bias=True):
    w_name = name + '_w'
    b_name = name + '_b'
    weight = weight_variable(shape, w_name)
    
    y = tf.nn.conv2d(x, weight, strides=stride, padding=pad)
    if use_bias is True:
        bias = bias_variable(shape[3], b_name)
        y = y + bias
    
    if activition == 'relu':
        y = relu(y)
    elif activition == 'lrelu':
        y = lrelu(y, alpha)
    elif activition == 'prelu':
        y = prelu(y, name+'prelu')
    #tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)
    return y


def deconvolution(x, scale, input_channels, padding='SAME'):
    shape = tf.shape(x)
    output_shape = [shape[0], scale * shape[1], scale * shape[2], 1]
    stride = [1, scale, scale, 1]

    weight = weight_variable(shape=[9, 9, 1, input_channels], name='weight_reconstruction')
    bias = bias_variable(shape=output_shape[-1], name='bias_reconstruction')

    output = tf.nn.conv2d_transpose(x, weight, output_shape, stride, padding=padding) + bias

    return output

def FSRCNN(image, scale):

    temp = conv2d(image, shape=[5, 5, 1, 56], activition='prelu', name='feature_extraction')

    temp = conv2d(temp, shape=[1, 1, 56, 16], activition='prelu', name='shrinking')

    temp = conv2d(temp, shape=[3, 3, 16, 12], activition='prelu', name='mapping1')
    temp = conv2d(temp, shape=[3, 3, 12, 12], activition='prelu', name='mapping2')
    temp = conv2d(temp, shape=[3, 3, 12, 12], activition='prelu', name='mapping3')
    temp = conv2d(temp, shape=[3, 3, 12, 12], activition='prelu', name='mapping4')

    temp = conv2d(temp, shape=[1, 1, 12, 56], activition='prelu', name='expanding')

    output = deconvolution(temp, scale, 56)

    return output
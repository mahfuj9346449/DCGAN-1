import tensorflow as tf
import numpy as np
import os

# Summaries
image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
summary_writer = tf.summary.FileWriter

# Batch normalization modification from 'r2rt.com/implementing-batch-normalization-in-tensorflow.html'
def batchnorm_wrapper(inputs, is_training, name, decay=0.99):
    with tf.variable_scope(name or 'bn' ):
        # Make batch norm variables for each channel
        scale = tf.get_variable('scale', [inputs.get_shape()[-1]], initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta', [inputs.get_shape()[-1]], initializer=tf.constant_initializer(0))
        pop_mean = tf.get_variable('pop_mean', [inputs.get_shape()[-1]], trainable=False, initializer=tf.constant_initializer(0))
        pop_var = tf.get_variable('pop_var', [inputs.get_shape()[-1]], trainable=False, initializer=tf.constant_initializer(1))

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs, [0,1,2]) # Calculate mean and variance for each channel
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1-decay))
            train_var = tf.assign(pop_var, pop_var * decay + pop_var*(1-decay))
            # Call population stats every time we calculate batch mean, batch var
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, variance_epsilon=1e-5)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, variance_epsilon=1e-5)


# Normal convolution(down sampling)
def conv2d(incoming, output_dim, filter_height=5, filter_width=5, stride_hor=2, stride_ver=2, name='downsamping'):
    with tf.variable_scope(name) :
        #  incoming.get_shape()[-1] : Number of filters(in_channels), # truncated_normal : values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked
        weights = tf.get_variable('weight', [filter_height, filter_width, incoming.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        # Convolution
        '''Padding : same
           out_height = ceil(float(in_height)/float(strides[1]))
           Padding : valid
           out_height = ceil(float(in_height-filter_height+1)/float(strides[1]))'''
        convolution = tf.nn.conv2d(incoming, weights, strides=[1,stride_hor, stride_ver, 1], padding='SAME') # Will have a shape : [batch, out_height, out_width, output_dim]
        biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0))
        weighted_sum = convolution+biases
#        print('Shape : %s' % (weighted_sum.get_shape()))
        return weighted_sum

# Fractional-strided(up sampling)
def deconv2d(incoming, output_shape, filter_height=4, filter_width=4, stride_hor=2, stride_ver=2, name='upsampling'):
    with tf.variable_scope(name):
        # Cautious with filter dimension : [height, widht, output_channels, input_channels], output_shape should be 1-D  array representing output shape
       weights = tf.get_variable('weight', [filter_height, filter_width, output_shape[-1], incoming.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
       deconvolution = tf.nn.conv2d_transpose(incoming, weights, output_shape=output_shape, strides=[1,stride_hor, stride_ver,1])
       biases = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0))
       weighted_sum = deconvolution+biases
       return weighted_sum

# incoming : [batch, hidden]
def linear(incoming, output_dim, name='linear'):
    with tf.variable_scope(name):
        weights = tf.get_variable('weight', [incoming.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0))
        weighted_sum = tf.matmul(incoming, weights) +biases
        return weighted_sum

# Used for discriminator
def lrelu(x, leak=0.2, name='leaky_relu'):
    return tf.maximum(x, x*leak)

if __name__ == "__main__":
    incoming = tf.get_variable('in', [1,4,4,1024])
    output_sh = [1,8,8,512]
    print(output_sh[-1])
    a = deconv2d(incoming, output_sh)
    print(a.get_shape())

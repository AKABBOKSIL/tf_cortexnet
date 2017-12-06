"""
    inference

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from tensorflow.contrib.layers import flatten
#from tflearn import dropout

import tensorflow as tf
import jy_tf_param as param
import jy_tf_nnLib as lib

def cortexnet18_cifar(x, phase_train):

    # init
    with tf.name_scope('conv1') as name:
        conv = lib.conv_layer( x, filter = param.filter, kernel = [7, 7], stride = 1 )
        conv_bn = lib.Relu(lib.Batch_Normalization(conv, training=phase_train, scope=name + '_bn'))

    with tf.name_scope('pool'):
        # 64 to 32
        pool = lib.Max_pooling(conv_bn, pool_size=[3, 3], stride = 1)

    # 본론#######################################################################################
    # next_input, sq1_fc1, fc_output
    filter = param.filter
    block_num = 2
    # block64, fc64 = cortexBlock(pool, filter,str(filter), block_num, phase_train)
    block64, fc64 = cortexBlock_1011(pool, filter, str(filter), block_num, phase_train)

    filter = filter * 2
    block_num = 2
    block128, fc128 = cortexBlock_1011(block64, filter, str(filter), block_num, phase_train)

    filter = filter * 2
    block_num = 2
    block256, fc256 = cortexBlock_1011(block128, filter, str(filter), block_num, phase_train)

    filter = filter * 2
    block_num = 2
    block512, fc512 = cortexBlock_1011(block256, filter, str(filter), block_num, phase_train)
    ########################################################################################

    with tf.name_scope('fc1') as name:
        size = block512.get_shape().as_list()[1]
        avg = lib.Avg_pooling(block512, pool_size=[size, size], stride=size, padding='SAME')
        layer = lib.Fully_connected(avg, 1000)
        #fc1 = lib.Relu(layer + fc64 + fc128 + fc256 + fc512)
        fc1_bn = lib.Relu(lib.Batch_Normalization(layer, phase_train, scope=name + '_fc'))

    with tf.name_scope('softmax'):
        py_x = flatten(fc1_bn)
        py_x = lib.Fully_connected(py_x, param.num_classes)

    return py_x



def cortexBlock(x, filter, scope, block_num, phase_train):
    """
    :param x: input data
    :param filter: convolutional filter size
    :param scope: block name
    :param block_num: # of blocks to apply
    :param phase_train: unusable
    :return:
    """
    next_input = 0

    for b in range(1, block_num + 1):
        if (b == 1):
            stride = 2
            input = x
        else:
            stride = 1
            input = next_input

        with tf.name_scope(scope + '-' + str(b)) as name:  # e.g. cr64-1
            input_conv1 = lib.conv_layer(input, filter=filter, kernel=[3, 3], stride=stride, activation=False)
            if (b == 1):
                x = input_conv1
            conv_bn = lib.Relu(lib.Batch_Normalization(input_conv1, training=phase_train, scope=name + '_conv'))

            max_pool = lib.Max_pooling(input, stride=stride)
            maxp_conv = lib.conv_layer(max_pool, filter=filter, kernel=[3, 3], activation=False)
            maxp_bn = lib.Batch_Normalization(maxp_conv, training=phase_train, scope=name + '_max')
            maxp_bn = lib.Relu(maxp_bn)

            avg_pool = lib.Avg_pooling(input, stride=stride)
            avgp_conv = lib.conv_layer(avg_pool, filter=filter, kernel=[3, 3], activation=False)
            avgp_bn = lib.Relu(lib.Batch_Normalization(avgp_conv, training=phase_train, scope=name + '_avg'))

            mixed_concat = lib.Concatenation([maxp_bn + conv_bn, avgp_bn + conv_bn])
            mixed_conv = lib.conv_layer(mixed_concat, filter=filter * 2, kernel=[3, 3], activation=True)
            mixed_bn = lib.Relu(lib.Batch_Normalization(mixed_conv, training=phase_train, scope=name + '_mixed'))

            if (b == block_num):
                next_input = mixed_bn
            else:
                shortcut_conv = lib.conv_layer(mixed_bn, filter=filter, kernel=[1, 1], activation=False)
                shortcut_bn = lib.Relu(
                    lib.Batch_Normalization(shortcut_conv, training=phase_train, scope=name + '_shortcut'))
                next_input = shortcut_bn

    with tf.name_scope(scope) as name:
        size = next_input.get_shape().as_list()[1]
        features = next_input.get_shape().as_list()[3]

        sq = lib.Avg_pooling(next_input, pool_size=[size, size], stride=size)
        sq_fc1 = lib.Relu(
            lib.Batch_Normalization(lib.Fully_connected(sq, units=1000), training=phase_train, scope=name + '_sq1'))
        sq_fc2 = lib.Sigmoid(lib.Batch_Normalization(lib.Fully_connected(sq_fc1, units=features), training=phase_train,
                                                     scope=name + '_sq2'))

        col = reshape = tf.reshape(sq_fc2, [-1, 1, 1, features])
        for i in range(size - 1):
            col = tf.concat([reshape, col], 1)
        sqzs = col
        for i in range(size - 1):
            sqzs = tf.concat([col, sqzs], 2)

        features = int(features / 2)
        excit, _ = tf.split(sqzs, [features, features], 3)
        fc_output = tf.cond(phase_train, lambda: lib.retFch(excit), lambda: lib.retZeros(excit))
        # fc_output = excit

    input += fc_output
    return next_input, sq_fc1



def cortexnet34(x, phase_train):
    # 이미지넷, 일반이미지(128x128) 학습 가능

    # init
    with tf.name_scope('conv1') as name:
        # 128 to 64
        conv = lib.conv_layer( x, filter = 64, kernel = [7, 7], stride = 2 )
        conv_bn = lib.Relu(lib.Batch_Normalization(conv, training=phase_train, scope=name + '_bn'))

    with tf.name_scope('pool'):
        # 64 to 32
        pool = lib.Max_pooling(conv_bn, pool_size=[3, 3], stride=2)

    # 본론#######################################################################################
    def cBlock(x, filter, scope, block_num, phase_train):
        """
        apply fch layer at every block unit
        = add fc_output to input layer at every block unit
        """
        next_input = 0
        # output_fc =0
        for b in range( 1, block_num + 1 ):
            if (b == 1):
                stride = 2
                input = x
            else:
                stride = 1
                input = next_input

            with tf.name_scope( scope + '-' + str( b ) ) as name:  # e.g. cr64-1
                input_conv1 = lib.conv_layer( input, filter = filter, kernel = [3, 3], stride = stride, activation = False )
                conv_bn = lib.Relu( lib.Batch_Normalization( input_conv1, training = phase_train, scope = name + '_conv' ) )

                max_pool = lib.Max_pooling( conv_bn, stride = 1 )
                maxp_conv = lib.conv_layer( max_pool, filter = filter, kernel = [1, 1], activation = False )
                maxp_bn = lib.Batch_Normalization( maxp_conv, training = phase_train, scope = name + '_max' )
                maxp_bn = lib.Relu( maxp_bn )

                avg_pool = lib.Avg_pooling( conv_bn, stride = 1 )
                avgp_conv = lib.conv_layer( avg_pool, filter = filter, kernel = [1, 1], activation = False )
                avgp_bn = lib.Relu( lib.Batch_Normalization( avgp_conv, training = phase_train, scope = name + '_avg' ) )

                mixed_concat = lib.Concatenation( [maxp_bn + conv_bn, avgp_bn + conv_bn] )
                mixed_conv = lib.conv_layer( mixed_concat, filter = filter, kernel = [3, 3], activation = False )
                mixed_bn = lib.Relu( lib.Batch_Normalization( mixed_conv, training = phase_train, scope = name + '_mixed' ) )
                next_input = mixed_bn

            with tf.name_scope( scope ) as name:
                size = next_input.get_shape().as_list()[1]
                features = next_input.get_shape().as_list()[3]

                sq = lib.Avg_pooling( next_input, pool_size = [size, size], stride = size )
                sq_fc1 = tf.nn.softmax( lib.Batch_Normalization( lib.Fully_connected( sq, units = param.num_classes ), training = phase_train,
                                             scope = name + '_sq1' ) )
                sq_fc2 = lib.Relu(lib.Batch_Normalization( lib.Fully_connected( sq_fc1, units =  features ), training = phase_train, scope = name + '_sq2' ) )

                excitation = tf.reshape( sq_fc2, [-1, 1, 1, features] )
                next_input = lib.Relu( excitation + input_conv1 )

        return next_input

    filter = param.filter
    block_num = 3
    block64 = cBlock(pool, filter, str(filter), block_num, phase_train)

    filter = filter * 2
    block_num = 4
    block128 = cBlock(block64, filter, str(filter), block_num, phase_train)

    filter = filter * 2
    block_num = 6
    block256 = cBlock(block128, filter, str(filter), block_num, phase_train)

    filter = filter * 2
    block_num = 3
    block512 = cBlock(block256, filter, str(filter), block_num, phase_train)
    ########################################################################################

    with tf.name_scope('fc1') as name:
        size = block512.get_shape().as_list()[1]
        avg = lib.Avg_pooling(block512, pool_size=[size, size], stride=size, padding='SAME')
        layer = lib.Fully_connected(avg, 1000)
        fc1_bn = lib.Relu(lib.Batch_Normalization(layer, phase_train, scope=name + '_fc'))

    with tf.name_scope('softmax'):
        py_x = flatten(fc1_bn)
        py_x = lib.Fully_connected(py_x, param.num_classes)

    return py_x

def cortexnet50(x, phase_train):
    # 이미지넷, 일반이미지(128x128) 학습 가능

    # init
    with tf.name_scope('conv1') as name:
        # 128 to 64
        if (param.picSize == 64):
            conv = lib.conv_layer(x, filter=64, kernel=[7, 7], stride=1)
        else:
            conv = lib.conv_layer(x, filter=64, kernel=[7, 7], stride=2)
        conv_bn = lib.Relu(lib.Batch_Normalization(conv, training=phase_train, scope=name + '_bn'))

    with tf.name_scope('pool'):
        # 64 to 32
        pool = lib.Max_pooling(conv_bn, pool_size=[3, 3], stride=2)

    # 본론#######################################################################################
    # next_input, sq1_fc1, fc_output
    filter = param.filter
    block_num = 3
    block64, fc64 = cortexBlock_50(pool, filter, str(filter), block_num, phase_train)

    filter = filter * 2
    block_num = 4
    block128, fc128 = cortexBlock_50(block64, filter, str(filter), block_num, phase_train)

    filter = filter * 2
    block_num = 6
    block256, fc256 = cortexBlock_50(block128, filter, str(filter), block_num, phase_train)

    filter = filter * 2
    block_num = 3
    block512, _ = cortexBlock_50(block256, filter, str(filter), block_num, phase_train)
    ########################################################################################

    with tf.name_scope('fc1') as name:
        size = block512.get_shape().as_list()[1]
        avg = lib.Avg_pooling(block512, pool_size=[size, size], stride=size, padding='SAME')
        layer = lib.Fully_connected(avg, 1000)
        fc1 = lib.Relu(layer + fc64 + fc128 + fc256)
        fc1_bn = lib.Relu(lib.Batch_Normalization(fc1, phase_train, scope=name + '_fc'))

    with tf.name_scope('softmax'):
        py_x = flatten(fc1_bn)
        py_x = lib.Fully_connected(py_x, param.num_classes)

    return py_x



def resnet34(x, phase_train):
    with tf.name_scope('conv1') as name:
        # 128 to 64
        conv = lib.conv_layer( x, filter = 64, kernel = [7, 7], stride = 2 )
        conv_bn = lib.Relu(lib.Batch_Normalization(conv, training=phase_train, scope=name + '_bn'))

    with tf.name_scope('pool'):
        # 64 to 32
        pool = lib.Max_pooling(conv_bn, pool_size=[3, 3], stride=2)

    # 본론#######################################################################################
    filter = param.filter  # filter = 64
    block64 = residual_block(pool, phase_train, filter, 3, str(filter), 1)

    filter = filter * 2
    block128 = residual_block(block64, phase_train, filter, 4, str(filter), 2)

    filter = filter * 2
    block256 = residual_block(block128, phase_train, filter, 6, str(filter), 2)

    filter = filter * 2
    block512 = residual_block(block256, phase_train, filter, 3, str(filter), 2)
    ########################################################################################

    with tf.name_scope('fc1') as name:
        size = block512.get_shape().as_list()[1]
        avg = lib.Avg_pooling(block512, pool_size=[size, size], stride=size, padding='SAME')
        fc1 = lib.Fully_connected(avg, 1000)
        fc1_bn = lib.Relu(lib.Batch_Normalization(fc1, phase_train, scope=name + '_fc'))

    with tf.name_scope('softmax'):
        py_x = flatten(fc1_bn)
        py_x = lib.Fully_connected(py_x, param.num_classes)

    return py_x


def residual_block(x, phase_train, filter, block_num, scope, strides):
    for b in range(1, block_num + 1):
        if (b == 1):
            stride = strides
            input = x
        else:
            stride = 1
            input = next_input

        with tf.name_scope('r' + scope + '-' + str(b)) as name:  # e.g. r128-1
            conv1 = lib.conv_layer(input, filter=filter, kernel=[3, 3], stride=stride, activation=False)
            conv_bn1 = lib.Relu(lib.Batch_Normalization(conv1, training=phase_train, scope=name + '_conv1'))

            conv2 = lib.conv_layer(conv_bn1, filter=filter, kernel=[3, 3], stride=1, activation=False)
            conv_bn2 = lib.Relu(lib.Batch_Normalization(conv2, training=phase_train, scope=name + '_conv2'))

        if (b == 1):
            next_input = tf.identity(conv_bn2)
        else:
            next_input = lib.Relu(conv_bn2 + tf.identity(input))

    output = next_input
    return output



def senet34(x, phase_train):
    """
    squeeze-excitation network
    https://arxiv.org/pdf/1709.01507.pdf
    """
    with tf.name_scope('conv1') as name:
        # 128 to 64
        conv = lib.conv_layer( x, filter = 64, kernel = [7, 7], stride = 2 )
        conv_bn = lib.Relu(lib.Batch_Normalization(conv, training=phase_train, scope=name + '_bn'))

    with tf.name_scope('pool'):
        # 64 to 32
        pool = lib.Max_pooling(conv_bn, pool_size=[3, 3], stride=2)

    def seBlock(x, phase_train, filter, block_num, scope, strides):

        for b in range( 1, block_num + 1 ):
            if (b == 1):
                stride = strides
                input = x
            else:
                stride = 1
                input = next_input

            with tf.name_scope( 'r' + scope + '-' + str( b ) ) as name:  # e.g. r128-1
                conv1 = lib.conv_layer( input, filter = filter, kernel = [3, 3], stride = stride, activation = False )
                conv_bn1 = lib.Relu(lib.Batch_Normalization( conv1, training = phase_train, scope = name + '_conv1' ) )

                conv2 = lib.conv_layer( conv_bn1, filter = filter, kernel = [3, 3], stride = 1, activation = False )
                conv_bn2 = lib.Relu(lib.Batch_Normalization( conv2, training = phase_train, scope = name + '_conv2' ) )

            with tf.name_scope( scope ) as name:  # e.g. cr64-1
                size = conv_bn2.get_shape().as_list()[1]
                features = conv_bn2.get_shape().as_list()[3]

                sq = lib.Avg_pooling( conv_bn2, pool_size = [size, size], stride = size )
                sq_fc1 = lib.Relu(lib.Batch_Normalization( lib.Fully_connected( sq, units = 1000 ), training = phase_train,scope = name + '_sq1' ) )
                sq_fc2 = lib.Sigmoid(lib.Batch_Normalization( lib.Fully_connected( sq_fc1, units = features ), training = phase_train,scope = name + '_sq2' ) )

                excitation = tf.reshape( sq_fc2, [-1, 1, 1, features] )
                scale = conv_bn2 * excitation
            if (b == 1):
                next_input = tf.identity( scale )
            else:
                next_input = lib.Relu( input + tf.identity( scale ) )

        return next_input

    # 본론#######################################################################################
    filter = param.filter  # filter = 64
    block64 = seBlock(pool, phase_train, filter, 3, str(filter), 1)

    filter = filter * 2
    block128 = seBlock(block64, phase_train, filter, 4, str(filter), 2)

    filter = filter * 2
    block256 = seBlock(block128, phase_train, filter, 6, str(filter), 2)

    filter = filter * 2
    block512 = seBlock(block256, phase_train, filter, 3, str(filter), 2)
    ########################################################################################

    with tf.name_scope('fc1') as name:
        size = block512.get_shape().as_list()[1]
        avg = lib.Avg_pooling(block512, pool_size=[size, size], stride=size, padding='SAME')
        fc1 = lib.Fully_connected(avg, 1000)
        fc1_bn = lib.Relu(lib.Batch_Normalization(fc1, phase_train, scope=name + '_fc'))

    with tf.name_scope('softmax'):
        py_x = flatten(fc1_bn)
        py_x = lib.Fully_connected(py_x, param.num_classes)

    return py_x

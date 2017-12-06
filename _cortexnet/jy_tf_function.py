from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import jy_tf_param as param
import os

"""to load image file from the disk"""
def read_image_from_disk(input_queue):
    #jpeg ver.
    label=input_queue[1]
    file_contents=tf.read_file(input_queue[0])
    rgb_image=tf.image.decode_jpeg(file_contents, channels=param.channel, name="decode_jpeg")
    rgb_image.set_shape([param.picSize, param.picSize, 3])

    return rgb_image, label

def read_image_png(input_queue):
    #png ver.
    label=input_queue[1]
    file_contents=tf.read_file(input_queue[0])
    rgb_image=tf.image.decode_png(file_contents, channels=param.channel, name="decode_png")
    rgb_image.set_shape([param.picSize, param.picSize, 3])

    return rgb_image, label

def read_images_from_list(image_list_file):
    f=open(image_list_file,'r')
    filenames=[]
    labels=[]

    arr=[0 for _ in range(param.num_classes)]

    for line in f:
        filename, label=line.split('\t')
        filenames.append(filename)

        arr[int(label)] = 1
        tmp = copy.deepcopy(arr)
        labels.append(tmp)
        arr[int(label)] = 0

    f.close()

    images=ops.convert_to_tensor(filenames,dtype=dtypes.string)
    img_labels=ops.convert_to_tensor(labels,dtype=dtypes.int32)
    input_queue=tf.train.slice_input_producer([images, img_labels], num_epochs=None, shuffle=True)

    if(param.imgType=='png'):
        image_list, label_list = read_image_png(input_queue)
    else:
        image_list, label_list = read_image_from_disk(input_queue)

    image_batch=tf.train.batch([image_list, label_list], batch_size=param.batch_size) #batch: 한번에 처리하는 사진의 장수

    return image_batch

def load_test_image(fileDir, index):
    files = os.listdir(fileDir)
    label = [0 for _ in range(param.num_classes)]
    label[index]=1

    imageQueue = tf.train.string_input_producer([(fileDir+'%s'%name) for name in files])
    _, content =tf.WholeFileReader().read(imageQueue)
    rgb_image=tf.image.decode_jpeg(content, channels=3)

    rgb_image.set_shape([param.picSize, param.picSize, 3])
    image_batch, image_label = tf.train.batch([rgb_image, label],
                                          batch_size=param.test_batchSize)
    return image_batch, image_label

def load_test_image2(dirTxt):
    """
    load test data at one time
    :param fileDir:
    :return:
    """

    f = open(dirTxt, 'r')
    filenames = []
    labels = []

    arr = [0 for _ in range(param.num_classes)]

    for line in f:
        filename, label = line.split('\t')
        filenames.append(filename)

        arr[int(label)] = 1
        tmp = copy.deepcopy(arr)
        labels.append(tmp)
        arr[int(label)] = 0

    f.close()

    images = ops.convert_to_tensor(filenames, dtype=dtypes.string)
    img_labels = ops.convert_to_tensor(labels, dtype=dtypes.int32)
    input_queue = tf.train.slice_input_producer([images, img_labels], num_epochs=None, shuffle=False)

    if (param.imgType == 'png'):
        image_list, label_list = read_image_png(input_queue)
    else:
        image_list, label_list = read_image_from_disk(input_queue)

    image_batch = tf.train.batch([image_list, label_list],
                                                batch_size=param.test_batchSize)

    return image_batch

def save_confusion_matrix(matrix):
    f = open(param.save_confusion, 'w')
    for row in range(param.num_classes):
        positive = 0
        for col in range(param.num_classes):
            f.write('%4d\t' %matrix[row][col])
            positive += matrix[row][col]
        f.write('\n')

    f.close()


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


import jy_tf_function as myFunc
import jy_tf_param as param
import jy_tf_model
import numpy as np

def test1():
    sess=tf.Session()
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))


    # Access saved Variables directly
    print(sess.run('bias:0'))
    # This will print 2, which is the value of bias that we saved


    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")
    feed_dict ={w1:13.0,w2:17.0}

    #Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

    print(sess.run(op_to_restore,feed_dict))
    #This will print 60 which is calculated

def test2():
    test_valid = myFunc.load_test_image2(param.testTxt)
    # pbTest = myFunc.load_test_image('E:/1.pear/TestSet_1/pb/', 0)

    x = tf.placeholder(tf.float32, [None, param.picSize, param.picSize, param.channel], name='X')
    #y_ = tf.placeholder(tf.float32, [None, param.num_classes], name='Y')

    phase_train = tf.placeholder(tf.bool, name='phase_train')
    model = getattr(jy_tf_model, param._model_)
    py_x = model(x, phase_train)

    #output = nnLib.training(py_x, y_)
    predict_op = tf.argmax(py_x, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run(feed_dict={phase_train: True})
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Restore model weights from previously saved model
        #saver = tf.train.import_meta_graph(param.save_saver + param.saver_name + '.meta')
        print(param.save_saver+ param.saver_name)
        saver.restore(sess, param.save_saver+ param.saver_name)
        print('restore')

        confusion_matrix = [[0 for _ in range(param.num_classes)] for _ in range(param.num_classes)]
        precision = 0
        iteration = int(param.num_classes * param.test_cnt / param.test_batchSize)
        for t in range(iteration):
            print(t)
            valid_tensor = sess.run(test_valid)
            pred = sess.run(predict_op, feed_dict={x: valid_tensor[0], phase_train: False})
            list = np.argmax(valid_tensor[1], axis=1)

            for c in range(len(pred)):
                confusion_matrix[int(pred[c])][int(list[c])] += 1
                if (int(pred[c]) == int(list[c])):
                    precision += 1

        for row in confusion_matrix:
            print(row)
        myFunc.save_confusion_matrix(confusion_matrix)
        print('accuracy: %.4f' % (precision / param.total_test_cnt))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    test2()
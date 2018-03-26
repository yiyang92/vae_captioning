########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################
# changed for image embedding task
import tensorflow as tf
import numpy as np

class vgg16:
    def __init__(self, imgs, weight_file=None, sess=None,
                 trainable_fe=False, trainable_top=False, dropout_keep=1.0):
        self.imgs = imgs
        self.dropout_keep = dropout_keep
        self.trainable_fe = trainable_fe
        self.trainable_top = trainable_top
        self.convlayers()
        self.fc_layers()
        if weight_file is not None and sess is not None:
            self.load_weights(weights, sess)

    def convlayers(self):
        self.parameters = []

        # zero-mean input, image_net weights mean
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939],
                               dtype=tf.float32, shape=[1, 1, 1, 3],
                               name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable(name='weights', shape=[3, 3, 3, 64],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases', shape=[64],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name='conv1_1')
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable(name='weights', shape=[3, 3, 64, 64],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases', shape=[64],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name='conv1_2')
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable(name='weights', shape=[3, 3, 64, 128],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases', shape=[128],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name='conv2_1')
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable(name='weights', shape=[3, 3, 128, 128],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases', shape=[128],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name='conv2_2')
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            kernel = tf.get_variable(name='weights', shape=[3, 3, 128, 256],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases', shape=[256],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name='conv3_1')
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.get_variable(name='weights', shape=[3, 3, 256, 256],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases', shape=[256],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name='conv3_2')
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.get_variable(name='weights', shape=[3, 3, 256, 256],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases', shape=[256],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name='conv3_3')
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
            kernel = tf.get_variable(name='weights', shape=[3, 3, 256, 512],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases', shape=[512],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name='conv4_1')
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases', shape=[512],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name='conv4_2')
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases', shape=[512],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name='conv4_3')
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            kernel = tf.get_variable(name='weights_conv', shape=[3, 3, 512, 512],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases_conv', shape=[512],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name='conv5_1')
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
            kernel = tf.get_variable(name='weights_conv', shape=[3, 3, 512, 512],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases_conv', shape=[512],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name='conv5_2')
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.get_variable(name='weights_conv', shape=[3, 3, 512, 512],
                                 trainable=self.trainable_fe)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='biases_conv', shape=[512],
                                 trainable=self.trainable_fe)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name='conv5_3')
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.variable_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.get_variable(name='weights', shape=[shape, 4096],
                                   trainable=self.trainable_top)
            fc1b = tf.get_variable(name='biases', shape=[4096],
                                   trainable=self.trainable_top)
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            if self.trainable_top:
                self.fc1 = tf.nn.dropout(self.fc1, self.dropout_keep)
            self.parameters += [fc1w, fc1b]
        # fc2
        with tf.variable_scope('fc2') as scope:
            fc2w = tf.get_variable(name='weights',shape=[4096, 4096],
                                   trainable=self.trainable_top)
            fc2b = tf.get_variable(name='biases', shape=[4096],
                                   trainable=self.trainable_top)
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            if self.trainable_top:
                self.fc2 = tf.nn.dropout(self.fc2, self.dropout_keep)
            self.parameters += [fc2w, fc2b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i == 30: # original file contains weights we dont need
                break
            sess.run(self.parameters[i].assign(weights[k]))
# vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

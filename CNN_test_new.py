# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import math
import numpy as np
import scipy.io as spio

#from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 128, 128, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 6])
  b_conv1 = bias_variable([6])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 6, 6])
  b_conv2 = bias_variable([6])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([32 * 32 * 6, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*6])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 128*128*3])
  b_fc2 = bias_variable([128*128*3])

  y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
"""
def loss(y_,y_conv):
  y_ = ((y_/255.0)-0.5)*2
  y_conv = ((y_conv/255.0)-0.5)*2
  mean_angle_error = 0
  for i in range(y_.shape[0]):
    prediction = y_conv[i,:]*y_conv[i,:]
    labels = y_[i,:]*y_[i,:]
    cross = y_conv[i,:]*y_[i,:]
    a11 = np.zeros((1,128*128))
    a12 = np.zeros((1,128*128))
    a22 = np.zeros((1,128*128))
    for j in range(128*128):
      a11[j] = prediction[j]+prediction[j+128*128]+prediction[j+2*128*128]
      a12[j] = cross[j]+cross[j+128*128]+cross[j+2*128*128]
      a22[j] = labels[j]+labels[j+128*128]+labels[j+2*128*128]
    cos_dist = a12/np.sqrt(a11*a22)
    cos_dist[np.isnan(cos_dist)] = -1
    cos_dist = np.clip(cos_dist,-1,1)
    angle_error = np.arccos(cos_dist)
    mean_angle_error += np.sum(angle_error)
  return mean_angle_error/y_.size/3
"""
      
def main(_):
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  #img_in = np.load('1.npz')
  # Create the model
  #dir_npz = [s for s in os.listdir() if os.path.splitext(s)[1]=='.npz']
  #dir_img = [x for x in os.listdir('train/color') if os.path.splitext(x)[1] =='.mat']
  x = tf.placeholder(tf.float32, [None, 16384])

  h = 128
  w = 128
  num_channels = 3
  batch_size = 25
  #isInitial = True

  # Define loss and optimizer
  #y_ = tf.placeholder(tf.float32, [None, 49152])

  #masks = tf.placeholder(tf.int32, [None, 49152])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  #cross_entropy = tf.reduce_mean(
  #    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  #cross_entropy = tf.reduce_mean(loss(y_,y_conv))
  #cross_entropy = tf.reduce_mean(tf.squared_difference(y_, y_conv))
  
  #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  
  #correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  
  #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  saver0 = tf.train.Saver()
  #labels = np.zeros((batch_size,h*w*num_channels))
  #imgs = np.zeros((batch_size,h*w))
  #masks_op = np.zeros((batch_size,h*w*num_channels))
  res_dict = {}
  
  with tf.Session() as sess:
    '''
    if isInitial:
      sess.run(tf.global_variables_initializer())
      isInitial = False
    else:
      saver0.restore(sess,'./model.ckpt')'''
    #sess.run(tf.global_variables_initializer())
    saver0.restore(sess,'./model.ckpt')
    #for i in range(20000):
    #choose = np.random.permutation(len(dir_img))
    #print(choose)
    #saver.save(sess,checkpoint_file)
    #saver.restore(sess,checkpoint_file)
    path_img = 'train/'+'test_img.mat'
    #path_masks = 'train/'+'test_mask.mat'
    img_in = spio.loadmat(path_img)
    #masks_in = spio.loadmat(path_masks)
    img_in_data = img_in['image']
    #masks_in_data = masks_in['mask_rec']
    batch = {'imgs':img_in_data}
    res = sess.run(y_conv, feed_dict={
          x: batch['imgs'], keep_prob: 1.0})
    print(res[0,:]==res[1,:])
    spio.savemat('./'+'normal'+'.mat',{"prediction":res})

    '''
    for i in range(math.ceil(len(dir_img)/batch_size)):
    #for i in range(5):
      #batch = mnist.train.next_batch(50)
      #batch = np.load(str(choose[i])+'.npz')
      
      for j in range(batch_size):
        name = choose[j+i*batch_size]
        path_img = 'train/color/'+str(name)+'.mat'
        path_labels = 'train/normal/'+str(name)+'.mat'
        path_masks = 'train/mask/'+str(name)+'.mat'
        img_in = spio.loadmat(path_img)
        labels_in = spio.loadmat(path_labels)
        masks_in = spio.loadmat(path_masks)

        img_in_data = img_in['image']
        labels_in_data = labels_in['gd_truth']
        masks_in_data = masks_in['mask']
        imgs[j,:] = img_in_data
        labels[j,:] = labels_in_data
        masks_op[j,:] = masks_in_data
      batch = {'imgs':imgs,'labels':labels,'masks':masks_op}
      #if i % 100 == 0:
      train_accuracy = cross_entropy.eval(feed_dict={
          x: batch['imgs'], y_: batch['labels'], masks:batch['masks'], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
      #test = y_.eval(feed_dict={
      #    x: batch['imgs'], y_: batch['labels'], keep_prob: 1.0})
      res = sess.run(y_conv, feed_dict={
          x: batch['imgs'], masks:batch['masks'], keep_prob: 1.0})
      
      #print(test.shape)
      train_step.run(feed_dict={x: batch['imgs'], y_: batch['labels'],masks:batch['masks'],keep_prob: 0.5})
    saver0.save(sess,'./model.ckpt')
    #spio.savemat('./'+str(i)+'.mat',{"prediction":res})
    #print('test accuracy %g' % accuracy.eval(feed_dict={
    #    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))'''
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/train/color',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

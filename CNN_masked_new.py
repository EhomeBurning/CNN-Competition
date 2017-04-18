from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import math
import numpy as np
import scipy.io as spio
import tensorflow as tf
import random

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
'''
def loss(y_,y_conv,masks):
  y_ = tf.multiply(tf.subtract(tf.divide(y_,255.0),0.5),2)
  num_pixels = tf.cast(tf.count_nonzero(masks),tf.float32)
  #y_ = tf.multiply(tf.subtract(y_,0.5),2)
  y_conv = tf.multiply(tf.subtract(tf.divide(y_conv,255.0),0.5),2)
  #y_conv = tf.multiply(tf.subtract(y_conv,0.5),2)
  mean_angle_error = tf.constant([0.0])
  prediction = tf.multiply(y_conv,y_conv)
  labels = tf.multiply(y_,y_)
  cross = tf.multiply(y_conv,y_)
  zeros_fil = tf.zeros_like(masks,tf.float32)
  ones_fil = tf.ones_like(masks,tf.int32)
  masks = tf.equal(ones_fil,masks)
  #y_conv = tf.where(masks,y_conv,zeros_fil)
  a11_1 = tf.zeros([25,128*128])
  a11_2 = tf.zeros([25,128*128])
  a11_3 = tf.zeros([25,128*128])
  a11 = tf.zeros([25,128*128])
  a11_1,a11_2,a11_3 = tf.split(prediction,3,1)
  a11 = tf.add(a11_1,tf.add(a11_2,a11_3))
  a11 = tf.where(masks,a11,zeros_fil)
    
  a12_1 = tf.zeros([25,128*128])
  a12_2 = tf.zeros([25,128*128])
  a12_3 = tf.zeros([25,128*128])
  a12 = tf.zeros([25,128*128])
  a12_1,a12_2,a12_3 = tf.split(cross,3,1)
  a12 = tf.add(a12_1,tf.add(a12_2,a12_3))
  a12 = tf.where(masks,a12,zeros_fil)
    
  a22_1 = tf.zeros([25,128*128])
  a22_2 = tf.zeros([25,128*128])
  a22_3 = tf.zeros([25,128*128])
  a22 = tf.zeros([25,128*128])
  a22_1,a22_2,a22_3 = tf.split(labels,3,1)
  a22 = tf.add(a22_1,tf.add(a22_2,a22_3))
  a22 = tf.where(masks,a22,zeros_fil)
  cos_dist = tf.div(a12,tf.sqrt(tf.multiply(a11,a22)))
    #cos_dist[np.isnan(cos_dist)] = -1
  cos_dist = tf.where(tf.is_nan(cos_dist),tf.negative(tf.ones_like(cos_dist)),cos_dist)
    #cos_dist = np.clip(cos_dist,-1,1)
  cos_dist = tf.clip_by_value(cos_dist,-1,1)
  angle_error = tf.acos(cos_dist)
  mean_angle_error = tf.add(mean_angle_error,tf.reduce_sum(angle_error))
  #num_pixels = tf.cast(tf.divide(num_pixels,3),tf.float32)
  return tf.divide(mean_angle_error,num_pixels)'''

      
def main(_):
  dir_img = [x for x in os.listdir('train/color') if os.path.splitext(x)[1] =='.mat']
  x = tf.placeholder(tf.float32, [None, 16384])
  h = 128
  w = 128
  num_channels = 3
  batch_size = 25

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 49152])
  y_conv, keep_prob = deepnn(x)
  masks = tf.placeholder(tf.int32, [None, 128*128])
  zeros_fil = tf.zeros_like(y_conv,tf.float32)
  ones_fil = tf.ones_like(masks,tf.int32)
  masks_use = tf.equal(ones_fil,masks)
  masks_use = tf.concat([masks_use,masks_use,masks_use],1)
  y_conv_masked = tf.where(masks_use,y_conv,zeros_fil)
  y_masked = tf.where(masks_use,y_,zeros_fil)
  cross_entropy = tf.reduce_mean(tf.squared_difference(y_masked, y_conv_masked))
  
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  saver0 = tf.train.Saver()
  labels = np.zeros((batch_size,h*w*num_channels))
  imgs = np.zeros((batch_size,h*w))
  masks_op = np.zeros((batch_size,h*w))
  res_dict = {}
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver0.restore(sess,'./model.ckpt')
    choose = np.random.permutation(len(dir_img))
    print(choose)
    for i in range(math.ceil(len(dir_img)/batch_size)):
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
      train_accuracy = cross_entropy.eval(feed_dict={
          x: batch['imgs'], y_: batch['labels'], masks:batch['masks'], keep_prob: 1.0})
      print('step %d, loss %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch['imgs'], y_: batch['labels'], masks:batch['masks'], keep_prob: 0.5})
      res = sess.run(y_conv, feed_dict={
          x: batch['imgs'], keep_prob: 1.0})
      a = random.randrange(25)
      b = random.randrange(25)
      print(a)
      print(b)
      #print(batch['imgs'][a,:]==batch['imgs'][b,:])
      print(np.array_equal(batch['imgs'][a,:],batch['imgs'][b,:]))
      #print(batch['imgs'][a,:].shape)
      #print(set(batch['imgs'][a,:])&set(batch['imgs'][b,:]))
      #print(batch['imgs'][a,64]==batch['imgs'][b,64])
      #print(batch['labels'][a,:]==batch['labels'][b,:])
      print(np.array_equal(batch['labels'][a,:],batch['labels'][b,:]))
      #print(np.array_equal(batch['masks'][a,:],batch['masks'][b,:]))      
      #print(batch['masks'][a,:]==batch['masks'][b,:])
      #print(res[a,:]==res[b,:])
      print(np.array_equal(res[a,:],res[b,:]))
    saver0.save(sess,'./model.ckpt')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/train/color',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

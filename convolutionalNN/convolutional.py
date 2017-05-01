from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import sklearn as sk

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


FLAGS = None
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(_):

  import conv_config as config

  # Import data
  #import the data from the training numpy array
  train_data = np.load(config.train_data)
  train_labels = np.load(config.train_labels)

  #shuffle the training data and labels
  train_data, train_labels = sk.utils.shuffle(train_data,train_labels,random_state = 5)

  #split the data 80/20
  train_data_train = train_data[:int(len(train_data)*0.8)]
  train_labels_train = train_labels[:int(len(train_data)*0.8)]
  train_data_validate = train_data[int(len(train_data)*0.8):]
  train_labels_validate = train_labels[int(len(train_data)*0.8):]


  #create an inverse logits ratio to scale the training to the number
  #of representations in the set
  bins, counts = np.unique(train_labels_train,return_counts=True)
  rep_sum = np.sum(counts)
  weight_vector = np.zeros(len(bins))

  for i in range(0,len(counts)):
    weight_vector[i] = 1-(counts[i]/rep_sum)

  # start a tensorflow seesion
  sess = tf.InteractiveSession()

  class_weights = tf.constant(weight_vector)

  #generate two placeholder arrays for training and labels
  x = tf.placeholder(tf.float32, shape=[None, len(train_data[0])])
  y_ = tf.placeholder(tf.float32, shape=[None,1])


  x_data = tf.reshape(x,[-1,len(train_data[0]),1,1])

  #let's start by doing a pooling layer because this data is huge
  pre_pool = tf.nn.max_pool(x_data, ksize=[1,config.pre_pool_size,1,1], strides=[1,config.pre_pool_stride,1,1], padding='SAME')


  #do the first convolutional layer
  W_conv1 = weight_variable([config.conv1_filter_size,1,1,config.conv1_num_filters])
  b_conv1 = bias_variable([config.conv1_num_filters])

  conv1 = tf.nn.conv2d(pre_pool, W_conv1, strides=[1,1,1,1], padding='SAME')
  conv1_out = tf.nn.relu(conv1 + b_conv1)

  #pool again every
  pool1 = tf.nn.max_pool(conv1_out, ksize=[1,config.pool1_size,1,1], strides=[1,config.pool1_stride,1,1], padding='SAME')

  #layer two convolution 100x32 - 64filters
  W_conv2 = weight_variable([config.conv2_filter_size,1,config.conv1_num_filters,config.conv2_num_filters])
  b_conv2 = bias_variable([config.conv2_num_filters])

  conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1,1,1,1], padding='SAME')
  conv2_out = tf.nn.relu(conv2 + b_conv2)

  #reduce by 4 just because it's devisible
  pool2 = tf.nn.max_pool(conv2_out, ksize=[1,config.pool2_size,1,1], strides=[1,config.pool2_stride,1,1], padding='SAME')

  #data size should now be 1x216x64 filters
  W_fc1 = weight_variable([(int(len(train_data[0])/config.pre_pool_size/config.pool1_size/config.pool2_size)*config.conv2_num_filters),config.hidden1_size])
  b_fc1 = bias_variable([config.hidden1_size])

  #I think the data should already be flat but this call is probably cheap
  pool2_flat = tf.reshape(pool2,[-1,int((len(train_data[0])/config.pre_pool_size/config.pool1_size/config.pool2_size)*config.conv2_num_filters)])

  #first hidden layer
  h1 = tf.nn.relu(tf.matmul(pool2_flat,W_fc1)+b_fc1)

  W_fc2 = weight_variable([config.hidden1_size,len(train_labels[0])])
  b_fc2 = bias_variable([len(train_labels[0])])

  #these are the outputs
  y_pre_weight = tf.matmul(h1,W_fc2)+b_fc2

  #weight the outputs to inverse class frequency
  y = tf.multiply(y_pre_weight,tf.cast(class_weights,tf.float32))

  #now calculate cross entropy on weighted inputs
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

  train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
  correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
  sess.run(tf.global_variables_initializer())

  for i in range(20000):
    #get a batch of 100 random training points from the training set
    batch_size = 100
    batch_nums = np.random.choice(len(train_data_train[:,0]),batch_size)
    sys.stdout.write("Batch {}".format(i))
    sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()

    #every 100th iteration let's calculate the train and validation accuracy
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:train_data_train, y_: train_labels_train})
      validation_accuracy = accuracy.eval(feed_dict={
          x:train_data_validate, y_: train_labels_validate})
      print("Step {}, Training accuracy: {}, Validation accuracy: {}".format(i, train_accuracy,validation_accuracy))

    #then train on the batch
    train_step.run(feed_dict={x: train_data_train[batch_nums], y_: train_labels_train[batch_nums]})

  #do something to evaluate test accuracy at the end
  #print("test accuracy %g"%accuracy.eval(feed_dict={
  #  x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

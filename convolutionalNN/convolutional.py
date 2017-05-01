from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


FLAGS = None
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial) 

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial) 

def main(_):
  # Import data
  #import the data from the training numpy array
  train_data = np.zeros((1,86400))
  train_labels = np.zeros((1,28))

  # start a tensorflow seesion
  sess = tf.InteractiveSession()

  #generate two placeholder arrays for training and labels
  x = tf.placeholder(tf.float32, shape=[None, len(train_data[0])])
  y_ = tf.placeholder(tf.float32, shape=[None,len(train_labels[0])])

  x_data = tr.reshape(x,[-1,len(train_data[0]),1])

  #let's start by doing a pooling layer because this data is huge
  pre_pool = tf.nn.max_pool(x_data, ksize=[1,10,1], strides=[1,10,1], padding='SAME')
  
  #try a convulational layer1 filter length of 100x32 filters
  W_conv1 = weight_variable([100,1,32])
  b_conv1 = bias_variable([32])

  conv1 = tf.nn.conv1d(pre_pool, W_conv1, strides=[1,1,1], padding='SAME')
  conv1_out = tf.nn.relu(conv1 + b_conv1)
  
  #pool again every 10 samples
  pool1 = tf.nn.max_pool(conv1_out, ksize=[1,10,1], strides=[1,10,1], padding='SAME')
  
  #layer two convolution 100x32 - 64filters
  W_conv2 = weight_variable([100,32,64])
  b_conv2 = bias_variable([64])

  conv2 = tf.nn.conv1d(pool1, W_conv2, strides=[1,1,1], padding='SAME')
  conv2_out = tf.nn.relu(conv2 + b_conv2)

  #reduce by 4 just because it's devisible
  pool2 = tf.nn.max_pool(conv2_out, ksize=[1,4,1], strides=[1,10,1], padding='SAME')
  
  #data size should now be 1x216x64 filters
  W_fc1 = weight_variable([216*64,1024])  
  b_fc1 = bias_variable([1024])

  #I think the data should already be flat but this call is probably cheap
  pool2_flat = tf.reshape(pool2,[-1,216*64])

  #first hidden layer
  h1 = tf.nn.relu(tf.matmul(pool2_flat,W_fc1)+b_fc1)

  W_fc2 = weight_variable([1024,len(train_labels[0])])  
  b_fc2 = bias_variable([len(train_labels[0])])
 
  #these are the outputs 
  y = tf.matmul(h1,W_fc2)+b_fc2
 
  #now we need to calculate the loss - this really should be weighted
  #to inverse class frequency 
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

  train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
  correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
  sess.run(tf.global_variables_initializer())

  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:batch[0], y_: batch[1]})
      print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import glob
import sklearn as sk
import sklearn.preprocessing as skp

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


FLAGS = None
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def main(_):
  import conv_config as config

  #instantiate a saver


  # Import data
  #import the data from the training numpy array
  train_data = np.load(config.train_data)
  train_data = train_data[:,:,0]
  train_labels = np.load(config.train_labels)
  train_ids = np.load(config.train_ids)
  unique_ids = np.unique(train_ids)
  num_ids = np.max(unique_ids)

  #print(np.max(train_data))
  #print(np.min(train_data))
  train_data = skp.normalize(train_data,axis=0)

  #shuffle the training data and labels
  train_data, train_labels = sk.utils.shuffle(train_data,train_labels,random_state = 5)

  #split the data 80/20
  train_data_train = train_data[:int(len(train_data)*0.8)]
  train_labels_train = train_labels[:int(len(train_data)*0.8)]
  train_ids_train = train_ids[:int(len(train_data)*0.8)]
  train_data_validate = train_data[int(len(train_data)*0.8):]
  train_labels_validate = train_labels[int(len(train_data)*0.8):]
  train_ids_validate = train_ids[int(len(train_data)*0.8):]


  #create an inverse logits ratio to scale the training to the number
  #of representations in the set
  bins, counts = np.unique(train_labels_train,return_counts=True)
  rep_sum = np.sum(counts)
  num_classes = len(bins)
  weight_vector = np.zeros(num_classes)

  id_to_label = np.zeros(num_ids+1)

  for i in range(0,num_ids+1):
    index = np.where(train_ids == i)
    if(len(index[0]) > 0):
      id_to_label[i] = train_labels[index[0][0]]

  id_to_lab = tf.constant(id_to_label)

  for i in range(0,len(counts)):
    weight_vector[i] = 1-(counts[i]/rep_sum)

  # start a tensorflow seesion
  sess = tf.InteractiveSession()

  class_weights = tf.constant(weight_vector)

  #generate two placeholder arrays for training and labels
  x = tf.placeholder(tf.float32, shape=[None, len(train_data[0])])
  y_ = tf.placeholder(tf.int32, shape=[None,1])
  ids = tf.placeholder(tf.int32, shape=[None,1])


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

  W_fc2 = weight_variable([config.hidden1_size,num_classes])
  b_fc2 = bias_variable([num_classes])

  #these are the outputs
  y = tf.matmul(h1,W_fc2)+b_fc2

  res = tf.argmax(y,1)

  #weight the outputs to inverse class frequency
  y_w = tf.multiply(y,tf.cast(class_weights,tf.float32))

  #now calculate cross entropy on weighted inputs
  y_ = tf.reshape(y_,[-1])
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=y_w))

  train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
  preds = tf.argmax(y,1)
  correct = tf.equal(preds,tf.cast(y_,tf.int64))
  accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
  
  one_hot_preds = tf.transpose(tf.one_hot(preds,num_classes))
  ids = tf.reshape(ids,[-1])
  one_hot_ids = tf.one_hot(ids,num_ids+1)
  votes = tf.matmul(one_hot_preds,one_hot_ids)
  votes = tf.transpose(votes)
  not_included = tf.not_equal(tf.reduce_max(votes,1),0)
  grouped_correct = tf.equal(tf.boolean_mask(tf.argmax(votes,1),not_included),tf.boolean_mask(id_to_lab,not_included))
  grouped_accuracy = tf.reduce_mean(tf.cast(grouped_correct,tf.float32))
  
  saver = tf.train.Saver()

  if(len(glob.glob(config.model_save_path + ".*")) > 0):
      print("Restoring model from checkpoint")
      saver.restore(sess, config.model_save_path)
  else:
      sess.run(tf.global_variables_initializer())
      print("No checkpoints found.")


  for i in range(20000):
    #get a batch of 100 random training points from the training set
    batch_size = 50
    batch_nums = np.random.choice(len(train_data_train[:,0]),batch_size)
    test_nums = np.random.choice(len(train_data_train[:,0]),2000)
    sys.stdout.write("Batch {}".format(i))
    sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()

    #every 100th iteration let's calculate the train and validation accuracy
    if i%100 == 0:
      train_accuracy, train_grouped,entropy = sess.run([accuracy,grouped_accuracy,cross_entropy],feed_dict={
          x:train_data_train[test_nums], y_: train_labels_train[test_nums], ids: train_ids_train[test_nums]})
      validation_accuracy, val_grouped = sess.run([accuracy,grouped_accuracy],feed_dict={
          x:train_data_train[test_nums], y_: train_labels_train[test_nums], ids:train_ids_validate})

      #train_accuracy = accuracy.eval(feed_dict={
      #    x:train_data_train[test_nums], y_: train_labels_train[test_nums]})
      #validation_accuracy = accuracy.eval(feed_dict={
      #    x:train_data_validate, y_: train_labels_validate})
      print("Step {}, Training accuracy: {}, Validation accuracy: {}".format(i, train_accuracy,validation_accuracy))
      print(cross_entropy.eval(feed_dict={x:train_data_train[test_nums], y_: train_labels_train[test_nums]}))
      
      saver.save(sess, config.model_save_path)

      #print("Step {}, Validation accuracy: {}".format(i, validation_accuracy))

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import glob
import sklearn as sk
import sklearn.preprocessing as skp
import imp

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
  config = imp.load_source('config',_[1])
  np.set_printoptions(threshold=np.nan,linewidth=200)
  #instantiate a saver

  # Import data
  #import the data from the training numpy array
  train_data = np.load(config.train_data)

  if(config.use_pf == True):
    train_data[:,:,0] = train_data[:,:,0]/4000
  else:
    train_data = train_data[:,:,0]
    train_data = train_data/4000


  train_labels = np.load(config.train_labels)
  train_ids = np.load(config.train_ids)

  test = False
  if(hasattr(config, 'test')):
    if(config.test):
      test = True
      test_data = np.load(config.test_data)
      if(config.use_pf == False):
        test_data = test_data[:,:,0]
        test_data = test_data/4000
      else:
        test_data[:,:,0] = test_data[:,:,0]/4000

      #test_data = skp.normalize(test_data,axis=0)
      test_labels = np.load(config.test_labels)
      test_ids = np.load(config.test_ids)

  unique_ids = np.unique(train_ids)
  num_ids = np.max(unique_ids)

  #print(np.max(train_data))
  #print(np.min(train_data))
  #std_scale = preprocessing.StandardScaler().fit(train_data)
  #train_data = std_scale.transform(train_data)
  #train_data = skp.normalize(train_data,axis=0)

  #shuffle the training data and labels
  train_data, train_labels, train_ids = sk.utils.shuffle(train_data,train_labels,train_ids,random_state = 5)

  #split the data 80/20
  train_data_train = train_data[:int(len(train_data)*0.8)]
  train_labels_train = train_labels[:int(len(train_data)*0.8)]
  train_ids_train = train_ids[:int(len(train_data)*0.8)]
  train_data_validate = train_data[int(len(train_data)*0.8):]
  train_labels_validate = train_labels[int(len(train_data)*0.8):]
  train_ids_validate = train_ids[int(len(train_data)*0.8):]

  id_to_label = np.zeros(num_ids+1)

  #make an id to label array for the training data
  for i in range(0,num_ids+1):
    index = np.where(train_ids == i)
    if(len(index[0]) > 0):
      id_to_label[i] = train_labels[index[0][0]]

  #make an id to label array for the test data if it exists
  if(hasattr(config, 'test')):
    if(config.test):
      for i in range(0,num_ids+1):
        index = np.where(test_ids == i)
        if(len(index[0]) > 0):
          id_to_label[i] = test_labels[index[0][0]]


  id_to_lab = tf.constant(id_to_label,dtype=tf.int64)

  #create an inverse logits ratio to scale the training to the number
  #of representations in the set
  bins, counts = np.unique(train_labels_train,return_counts=True)
  rep_sum = np.sum(counts)
  num_classes = len(bins)
  weight_vector = np.zeros(num_classes)

  for i in range(0,len(counts)):
    weight_vector[i] = (1/num_classes)/counts[i]

  probability_vector = np.zeros(len(train_data_train))

  for i in range(0,len(probability_vector)):
    probability_vector[i] = weight_vector[train_labels_train[i]]

  # start a tensorflow seesion
  sess = tf.InteractiveSession()

  class_weights = tf.constant(weight_vector,dtype=tf.float32)

  #generate two placeholder arrays for training and labels
  if(config.use_pf == True):
    x = tf.placeholder(tf.float32, shape=[None, len(train_data[0]),2])
    x_data = tf.reshape(x,[-1,len(train_data[0]),2,1])
  else:
    x = tf.placeholder(tf.float32, shape=[None, len(train_data[0])])
    x_data = tf.reshape(x,[-1,len(train_data[0]),1,1])

  y_ = tf.placeholder(tf.int32, shape=[None,1])
  ids = tf.placeholder(tf.int32, shape=[None,1])

  #let's start by doing a pooling layer because this data is huge
  lpre_pool = tf.nn.max_pool(x_data, ksize=[1,config.lpre_pool_size,1,1], strides=[1,config.lpre_pool_stride,1,1], padding='SAME')

  #do the first convolutional layer
  if(config.use_pf == True):
    lW_conv1 = weight_variable([config.lconv1_filter_size,2,1,config.lconv1_num_filters])
  else:
    lW_conv1 = weight_variable([config.lconv1_filter_size,1,1,config.lconv1_num_filters])

  lb_conv1 = bias_variable([config.lconv1_num_filters])

  lconv1 = tf.nn.conv2d(lpre_pool, lW_conv1, strides=[1,1,1,1], padding='SAME')
  lconv1_out = tf.nn.relu(lconv1 + lb_conv1)

  #pool again every
  lpool1 = tf.nn.max_pool(lconv1_out, ksize=[1,config.lpool1_size,1,1], strides=[1,config.lpool1_stride,1,1], padding='SAME')


  #let's start by doing a pooling layer because this data is huge
  pre_pool = tf.nn.max_pool(x_data, ksize=[1,config.pre_pool_size,1,1], strides=[1,config.pre_pool_stride,1,1], padding='SAME')

  #do the first convolutional layer
  if(config.use_pf == True):
    W_conv1 = weight_variable([config.conv1_filter_size,2,1,config.conv1_num_filters])
  else:
    W_conv1 = weight_variable([config.conv1_filter_size,1,1,config.conv1_num_filters])

  b_conv1 = bias_variable([config.conv1_num_filters])

  conv1 = tf.nn.conv2d(pre_pool, W_conv1, strides=[1,1,1,1], padding='SAME')
  conv1_out = tf.nn.relu(conv1 + b_conv1)

  #pool again every
  pool1 = tf.nn.max_pool(conv1_out, ksize=[1,config.pool1_size,1,1], strides=[1,config.pool1_stride,1,1], padding='SAME')

  #layer two convolution 100x32 - 64filters
  if(config.use_pf == True):
    W_conv2 = weight_variable([config.conv2_filter_size,2,config.conv1_num_filters,config.conv2_num_filters])
  else:
    W_conv2 = weight_variable([config.conv2_filter_size,1,config.conv1_num_filters,config.conv2_num_filters])

  b_conv2 = bias_variable([config.conv2_num_filters])

  conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1,1,1,1], padding='SAME')
  conv2_out = tf.nn.relu(conv2 + b_conv2)

  #reduce by 4 just because it's devisible
  pool2 = tf.nn.max_pool(conv2_out, ksize=[1,config.pool2_size,1,1], strides=[1,config.pool2_stride,1,1], padding='SAME')

  if(config.use_pf == True):
    pool2_size = (int((len(train_data[0])*2)/config.pre_pool_size/config.pool1_size/config.pool2_size)*config.conv2_num_filters)
    lpool2_size = (int(((len(train_data[0])*2)/config.lpre_pool_size/config.lpool1_size)*config.lconv1_num_filters))
  else:
    pool2_size = (int((len(train_data[0]))/config.pre_pool_size/config.pool1_size/config.pool2_size)*config.conv2_num_filters)
    lpool2_size = (int(((len(train_data[0]))/config.lpre_pool_size/config.lpool1_size)*config.lconv1_num_filters))


  pool2_flat = tf.reshape(pool2,[-1,pool2_size])
  lpool2_flat = tf.reshape(lpool1,[-1,lpool2_size])

  combined_flat = tf.concat([pool2_flat,lpool2_flat],1)

  #data size should now be 1x216x64 filters
  W_fc1 = weight_variable([pool2_size+lpool2_size,config.hidden1_size])
  b_fc1 = bias_variable([config.hidden1_size])

  #first hidden layer
  h1 = tf.nn.relu(tf.matmul(combined_flat,W_fc1)+b_fc1)

  keep_prob = tf.constant(config.keep_prob,dtype=tf.float32)
  h1_drop = tf.nn.dropout(h1,keep_prob)

  W_fc2 = weight_variable([config.hidden1_size,num_classes])
  b_fc2 = bias_variable([num_classes])

  #these are the outputs
  y = tf.matmul(h1_drop,W_fc2)+b_fc2


  #weight the outputs to inverse class frequency
  #y_w = tf.multiply(y,tf.cast(class_weights,tf.float32))
  y_w = y

  #now calculate cross entropy on weighted inputs
  y_ = tf.reshape(y_,[-1])
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=y_w))

  train_step = tf.train.AdamOptimizer(config.lr).minimize(cross_entropy)
  preds = tf.argmax(y,1)
  correct = tf.equal(preds,tf.cast(y_,tf.int64))
  accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

  #this stuff allows us to calaculate device grouped accuracy
  one_hot_preds = tf.transpose(tf.one_hot(preds,num_classes))
  ids = tf.reshape(ids,[-1])
  one_hot_ids = tf.one_hot(ids,num_ids+1)
  votes = tf.matmul(one_hot_preds,one_hot_ids)
  votes = tf.transpose(votes)
  good_votes = tf.reduce_max(votes,1)
  not_included = tf.not_equal(good_votes,0)
  voted_labels = tf.argmax(votes,1)
  filtered_votes = tf.boolean_mask(voted_labels,not_included)
  filtered_labels = tf.boolean_mask(id_to_lab,not_included)
  grouped_correct = tf.equal(filtered_votes,filtered_labels)
  grouped_accuracy = tf.reduce_mean(tf.cast(grouped_correct,tf.float32))

  #this allows us to show a confusion matrix
  one_hot_voted_labels = tf.one_hot(voted_labels,num_classes)
  zone_hot_voted_labels = tf.where(not_included,one_hot_voted_labels,tf.cast(tf.zeros((num_ids+1,num_classes)),tf.float32))
  one_hot_votes = tf.transpose(zone_hot_voted_labels)
  one_hot_labels = tf.one_hot(id_to_lab,num_classes)
  confusion_matrix = tf.matmul(one_hot_votes,one_hot_labels)

  #this allows us to show a day-by-day confusion matrix
  day_confusion_matrix = tf.matmul(tf.transpose(votes),one_hot_labels)

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
    batch_nums = np.random.choice(len(train_data_train[:,0]),batch_size,p=probability_vector)
    test_nums = np.random.choice(len(train_data_train[:,0]),2000)
    sys.stdout.write("Batch {}".format(i))
    sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()

    #every 100th iteration let's calculate the train and validation accuracy
    if i%100 == 0:
      train_accuracy, train_grouped, entropy, train_confusion = sess.run([accuracy,grouped_accuracy,cross_entropy,confusion_matrix],
                  feed_dict={x:train_data_train[test_nums], y_: train_labels_train[test_nums], ids: train_ids_train[test_nums]})
      validation_accuracy, val_grouped,val_confusion = sess.run([accuracy,grouped_accuracy,confusion_matrix],
                  feed_dict={x:train_data_validate, y_: train_labels_validate, ids:train_ids_validate})

      print("Step {}, Training accuracy: {}, Validation accuracy: {}, Cross Entropy: {}".format(i,
                                                                                 train_accuracy,validation_accuracy,entropy))
      print("Grouped Training accuracy: {}, Grouped Validation accuracy: {}".format(train_grouped,val_grouped))
      print("")

      print("Training conufsion matrix")
      print(train_confusion)
      print("Validation conufsion matrix")
      print(val_confusion)

      saver.save(sess, config.model_save_path)

      if(test):
        test_accuracy, test_grouped, test_confusion = sess.run([accuracy,grouped_accuracy,confusion_matrix],
                      feed_dict={x:test_data, y_: test_labels, ids: test_ids})
        print("Test accuracy: {}, test accuracy grouped: {}".format(test_accuracy,test_grouped))
        print(test_confusion)

    #then train on the batch
    train_step.run(feed_dict={x: train_data_train[batch_nums], y_: train_labels_train[batch_nums]})


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='./conv_config.py',
                      help='Configuration file')
  FLAGS, unparsed = parser.parse_known_args()
  args = parser.parse_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + [args.config])

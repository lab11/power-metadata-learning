"""
Adapted from https://github.com/RobRomijnders/LSTM_tsc
"""

"""
LSTM for time series classification

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

import numpy as np
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn.python.ops import core_rnn

def load_data(direc,ratio):
  """Input:
  direc: location of the data sets
  ratio: ratio to split training and testset
  """
  data_train = np.load(direc+'/train.npy')[:,:,0]
  label_train = np.load(direc+'/trainLabels.npy')
  #data_unseen = np.load(direc+'unseen')
  #label_unseen = np.load(direc+'unseenLabel')
  N = data_train.shape[0]

  ratio = (ratio*N).astype(np.int32)
  ind = np.random.permutation(N)
  X_train = data_train[ind[:ratio[0]]]
  X_val =   data_train[ind[ratio[0]:ratio[1]]]
  X_test =  data_train[ind[ratio[1]:]]
  print(X_train.shape)
  print(X_val.shape)
  print(X_test.shape)
  # Targets have labels 1-indexed. We subtract one for 0-indexed
  y_train = label_train[ind[:ratio[0]]]
  y_val =   label_train[ind[ratio[0]:ratio[1]]]
  y_test =  label_train[ind[ratio[1]:]]
  print(y_train.shape)
  print(y_val.shape)
  print(y_test.shape)
  return X_train,X_val,X_test,y_train,y_val,y_test


def sample_batch(X_train,y_train,batch_size):
  """ Function to sample a batch for training"""
  N,data_len = X_train.shape
  ind_N = np.random.choice(N,batch_size,replace=False)
  X_batch = X_train[ind_N]
  y_batch = y_train[ind_N]
  return X_batch,y_batch


class Model():
  def __init__(self,config):

    num_layers = config['num_layers']
    hidden_size = config['hidden_size']
    max_grad_norm = config['max_grad_norm']
    self.batch_size = config['batch_size']
    num_val = config['num_val']
    learning_rate = config['learning_rate']
    num_classes = config['num_classes']
    weight_vec = config['weight_vec']
    """Place holders"""
    self.input = tf.placeholder(tf.float32, [None, num_val], name = 'input')
    self.labels = tf.placeholder(tf.int64, [None], name='labels')
    self.keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')

    print('Building Pooling layer')
    with tf.name_scope("Pooling") as scope:
        pool_input = tf.reshape(self.input,[-1,num_val,1,1], name = 'pool_input')
        pre_pool = tf.nn.max_pool(pool_input, ksize=[1,config['pre_pool_size'],1,1], strides=[1,config['pre_pool_stride'],1,1], padding='SAME', name='pre_pool')
        pre_pool = tf.squeeze(pre_pool, [2,3], name='pre_pool_reshape')

    print('Building LSTM cells')
    with tf.name_scope("LSTM_setup") as scope:
      def single_cell():
        return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size),output_keep_prob=self.keep_prob)

      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
      self.initial_state = cell.zero_state(self.batch_size, tf.float32)

    inputs_expand = tf.expand_dims(pre_pool,axis=2)
    outputs,_ = tf.nn.dynamic_rnn(cell, inputs_expand, swap_memory=True,dtype=tf.float32)
    output = tf.transpose(outputs, [1, 0, 2])[-1]

    print('Building Loss and Classification')
    #Generate a classification from the last cell_output
    #Note, this is where timeseries classification differs from sequence to sequence
    #modelling. We only output to Softmax at last time step
    with tf.name_scope("Softmax") as scope:
      with tf.variable_scope("Softmax_params"):
        softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
        softmax_b = tf.get_variable("softmax_b", [num_classes])
      class_weights = tf.constant(weight_vec)
      logits = tf.multiply(tf.nn.xw_plus_b(output, softmax_w, softmax_b), tf.cast(class_weights, tf.float32))
      #Use sparse Softmax because we have mutually exclusive classes
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.labels,name = 'softmax')
      self.cost = tf.reduce_sum(loss) / self.batch_size
    with tf.name_scope("Evaluating_accuracy") as scope:
      correct_prediction = tf.equal(tf.argmax(logits,1),self.labels)
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      h1 = tf.summary.scalar('accuracy',self.accuracy)
      h2 = tf.summary.scalar('cost', self.cost)


    print('Building Optimizer')
    """Optimizer"""
    with tf.name_scope("Optimizer") as scope:
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),max_grad_norm)   #We clip the gradients to prevent explosion
      optimizer = tf.train.AdamOptimizer(learning_rate)
      gradients = zip(grads, tvars)
      self.train_op = optimizer.apply_gradients(gradients)
      # Add histograms for variables, gradients and gradient norms.
      # The for-loop loops over all entries of the gradient and plots
      # a histogram. We cut of
      #for gradient, variable in gradients:  #plot the gradient of each trainable variable
      #       if isinstance(gradient, ops.IndexedSlices):
      #         grad_values = gradient.values
      #       else:
      #         grad_values = gradient

      #       tf.summary.histogram(variable.name, variable)
      #       tf.summary.histogram(variable.name + "/gradients", grad_values)
      #       tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

    #Final code for the TensorBoard
    self.merged = tf.summary.merge_all()
    self.init_op = tf.global_variables_initializer()
    self.saver = tf.train.Saver()
    print('Finished computation graph')


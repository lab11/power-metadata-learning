"""
LSTM for time series classification

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from datetime import datetime
import configparser
import numpy as np
import tensorflow as tf  #TF 1.1.0rc1
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
from lstm_model import Model,sample_batch,load_data

config = configparser.ConfigParser()
config.read('config.ini')
hparams = config['Hyperparameters']

#Set these directories
dataset_config = config['Datasets']
logging_config = config['Logging']
direc = dataset_config['directory']
log_dir = logging_config['log_directory']
model_path = logging_config['model_path'] + '/model'
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Generate run name
start_time = str(datetime.now()).replace(" ", "_")
param_dir = log_dir + '/' + hparams['num_layers'] + '_' + hparams['hidden_size'] + '_' + hparams['max_grad_norm'] + '_' + hparams['batch_size'] + '_' + hparams['learning_rate']
run_dir = param_dir + '/' + start_time

if not os.path.exists(param_dir):
    os.makedirs(param_dir)
os.makedirs(run_dir)


"""Load the data"""
ratio = np.array([float(dataset_config['train_ratio']), float(dataset_config['validate_ratio'])]) #Ratios where to split the training and validation set
X_train,X_val,X_test,y_train,y_val,y_test = load_data(direc,ratio)
N,num_val = X_train.shape
bins, counts = np.unique(y_train, return_counts=True)
num_classes = len(bins)
print('classes: {}'.format(num_classes))

"""Calculate Loss Weight Vector"""
weight_vec = 1 - counts/N

"""Hyperparamaters"""
max_iterations = int(hparams['max_iterations'])
dropout = float(hparams['dropout'])
batch_size = int(hparams['batch_size'])
model_config = {'num_layers' :      int(hparams['num_layers']),    #number of layers of stacked RNN's
                'hidden_size' :     int(hparams['hidden_size']),   #memory cells in a layer
                'max_grad_norm' :   int(hparams['max_grad_norm']), #maximum gradient norm during training
                'pre_pool_size':    int(hparams['pre_pool_size']),
                'pre_pool_stride':  int(hparams['pre_pool_stride']),
                'batch_size' :      batch_size,
                'learning_rate' :   float(hparams['learning_rate']),
                'num_val':          num_val,
                'num_classes':      num_classes,
                'weight_vec':      weight_vec}

epochs = np.floor(batch_size*max_iterations / N)
print('Train %.0f samples in approximately %d epochs' %(N,epochs))

#Instantiate a model
model = Model(model_config)
saver = model.saver

"""Session time"""
sess = tf.Session() #Depending on your use, do not forget to close the session
writer = tf.summary.FileWriter(run_dir, sess.graph)  #writer for Tensorboard
if os.path.isfile(model_path+".index"):
    saver.restore(sess, model_path)
    print("Load Model")
else:
    sess.run(model.init_op)

cost_train_ma = -np.log(1/float(num_classes)+1e-9)  #Moving average training cost
acc_train_ma = 0.0
try:
  for i in range(max_iterations):
    print('iterations: {}'.format(i))
    X_batch, y_batch = sample_batch(X_train,y_train,batch_size)

    #Next line does the actual training
    cost_train, acc_train,_ = sess.run([model.cost,model.accuracy, model.train_op],feed_dict = {model.input: X_batch,model.labels: y_batch,model.keep_prob:dropout})
    cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
    acc_train_ma = acc_train_ma*0.99 + acc_train*0.01
    if i%10 == 1:
    #Evaluate validation performance
      X_batch, y_batch = sample_batch(X_val,y_val,batch_size)
      cost_val, summ,acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
      print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,max_iterations,cost_train,cost_val,cost_train_ma,acc_train,acc_val,acc_train_ma))
      #Write information to TensorBoard
      writer.add_summary(summ, i)
      writer.flush()
      #Save model
      saver.save(sess, model_path)

except KeyboardInterrupt:
  pass

#Save model
saver.save(sess, model_path)

epoch = float(i)*batch_size/N
print('Trained %.1f epochs, accuracy is %5.3f and cost is %5.3f'%(epoch,acc_val,cost_val))

#now run in your terminal:
# $ tensorboard --logdir = <log_dir>




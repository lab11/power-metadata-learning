#the goal of this file is to take a tab separated output file from the SQL database
#then store it in a numpy array.

#it should be cleaned such that data appears every second

#it will be stored such that data is 24 hour aligned

import numpy as np
import sys
import os
import argparse
from datetime import datetime, date
import subprocess
import glob
import matplotlib.pyplot as plt

SEC_IN_DAY = 24*60*60

unseenTestRatio = .2

parser = argparse.ArgumentParser(description='Process data input files')
parser.add_argument('inputdir', metavar='I', type=str, nargs='+',
                    help='Input directory with npy array files')
parser.add_argument('labelFile', metavar='O', type=str, nargs='+',
                    help='A file with a comma separated list of labels')

args = parser.parse_args()

def split_data(data):
        splits = np.arange(SEC_IN_DAY, data.shape[0], SEC_IN_DAY)
        days = np.array(np.split(data, splits))
        return days


#read the labels file and make a dict of lists
labelFile = open(args.labelFile[0],'r')
labels = labelFile.readline().strip().split(',')

labelToFilenames = {}
labelToData = {}
for label in labels:
    labelToFilenames[label] = []
    labelToData[label] = []

#now get a list of data files
dataFiles = os.listdir(args.inputdir[0])

for dataFile in dataFiles:
    label = dataFile.split('_')[1].split('.')[0]
    if(label in labelToFilenames):
        labelToFilenames[label].append(dataFile)
    else:
        print("Warning: No label for {}".format(dataFile))

print("Found following data files:")

for key in labelToFilenames:
    numFiles = len(labelToFilenames[key])
    print("{}: {} files".format(key,numFiles))

print("\nLoading data and partitioning by day")
for key in labelToFilenames:
    print('working on ' + key)
    for filename in labelToFilenames[key]:
        data = split_data(np.load(args.inputdir[0] + '/' + filename))
        #percentile = np.percentile(data[:,:,0], 90, 1)
        #maximum = np.amax(data[:,:,0], 1)
        #mean = np.mean(data[:,:,0], 1)
        #keep = maximum > 3
        labelToData[key].append(data)

print('\nGenerate unseen set')

labelToUnseen = {}
labelToTrain = {}
for key in labelToFilenames:
    numFiles = len(labelToFilenames[key])
    if numFiles <= 2:
        print('No files for label ' + key + ', skipping...')
        continue
    print('Attempting to partition ' + key)
    # load devices
    numPoints = 0
    devices = labelToData[key]
    for data in devices:
        numPoints += data.shape[0]*data.shape[1]
    numTries = 0
    while numTries < 1000:
        pick = np.random.choice(numFiles, int(np.ceil(numFiles * unseenTestRatio)), False)
        unseenDevices = [devices[i] for i in pick]
        unseenNumPoints = 0
        for device in unseenDevices:
            unseenNumPoints += device.shape[0] * device.shape[1]
        print(key + 'pick represents {:.2g}'.format(unseenNumPoints/numPoints))
        if unseenNumPoints/numPoints > (unseenTestRatio - .1) and \
            unseenNumPoints/numPoints < (unseenTestRatio + .1):
            labelToUnseen[key] = unseenDevices
            labelToTrain[key] = [devices[i] for i in set(range(len(devices))) - set(pick)]

            break;
        numTries += 1
    if key not in labelToUnseen:
        print('Failed to choose unseen device(s) for ' + key)
        for i,device in enumerate(labelToData[key]):
            print("device {} consists of {:.2}".format(i, device.shape[0]*device.shape[1]/numPoints))
            print("Choices: ")
            choices = [int(i) for i in input().split()]
            labelToUnseen[key] = [devices[i] for i in choices]

# Need to concatenate the two sets, preserving each set's labels
# Need to split labelToTrain into training and validation
# Also need to generate class weight vector for loss function

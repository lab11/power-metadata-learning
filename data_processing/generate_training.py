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

unseenTestRatio = .2

parser = argparse.ArgumentParser(description='Process data input files')
parser.add_argument('inputdir', metavar='I', type=str, nargs='+',
                    help='Input directory with npy array files')
parser.add_argument('labelFile', metavar='O', type=str, nargs='+',
                    help='A file with a comma separated list of labels')

args = parser.parse_args()


#read the labels file and make a dict of lists
labelFile = open(args.labelFile[0],'r')
labels = labelFile.readline().strip().split(',')

labelToFilenames = {}
for label in labels:
    labelToFilenames[label] = []

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


print('\nGenerate unseen set')

labelToUnseen = {}
labelToTest = {}
for key in labelToFilenames:
    numFiles = len(labelToFilenames[key])
    if(numFiles <= 2):
        print('No files for label ' + key + ', skipping...')
        continue
    numPoints = 0

    # load devices
    devices = []
    for filename in labelToFilenames[key]:
        device = np.load(args.inputdir[0] + '/' + filename)
        devices.append(device)
        numPoints += device.shape[0]
    numTries = 0
    while numTries < 10:
        pick = np.random.choice(numFiles, int(np.ceil(numFiles * unseenTestRatio)), False)
        unseenDevices = list(devices[i] for i in pick)
        unseenNumPoints = 0
        for device in unseenDevices:
            unseenNumPoints += device.shape[0]
        print(unseenNumPoints)
        print(numPoints)
        print()
        if unseenNumPoints/numPoints > (unseenTestRatio - .1) and \
            unseenNumPoints/numPoints < (unseenTestRatio + .1):
            labelToUnseen[key] = unseenDevices
            labelToTest[key] = devices
            for ind in pick:
                del labelToTest[key][ind]
            break;
        numTries += 1
    if len(labelToUnseen[key]) == 0:
        print('Failed to choose unseen device(s)')
        exit()
    #numTrain = int((1-unseenTestRatio)*numFiles)
    #trainDict[key] = labelToFilenames[key][:numTrain]
    #unseenTestDict[key] = labelToFilenames[key][numTrain:]

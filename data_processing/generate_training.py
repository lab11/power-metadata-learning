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

SEC_IN_DAY = 60*60

seenTestRatio = .2
unseenTestRatio = .2

largeLabels = ['Blender', 'Blowdryer', 'CableBox', 'CoffeeMaker',
        'CurlingIronStraightener', 'Light',
        'Fan', 'Refrigerator', 'LaptopComputer', 'Microwave',
        'RouterModemSwitch', 'PhoneCharger', 'Television', 'Toaster']
smallLabels = ['Television', 'Refrigerator', 'Microwave', 'LaptopComputer',
        'CableBox', 'PhoneCharger', 'Toaster', 'CoffeeMaker', 'Light']

houseUniqueID = [4, 16, 21, 35, 61, 62, 82, 208, 104, 105, 106, 107, 109, 169, 231, 233, 219, 232, 215, 190, 191, 196, 257, 264, 276, 132, 133, 151, 152, 149, 150, 125, 135]

parser = argparse.ArgumentParser(description='Process data input files')
parser.add_argument('inputdir', metavar='I', type=str,
                    help='Input directory with npy array files')
parser.add_argument('labelFile', metavar='L', type=str,
                    help='A file with a comma separated list of labels')
parser.add_argument('outputdir', metavar='O', type=str,
                    help='Directory to output training and unseen npy array files')
parser.add_argument('-size', type=str, default='full', help="Use a 'large', 'small', or 'full' label set")
parser.add_argument('--house', dest='house', action='store_true', help="Use a single house deployment as the unique set")

args = parser.parse_args()

def split_data(data):
        splits = np.arange(SEC_IN_DAY, data.shape[0], SEC_IN_DAY)
        days = np.array(np.split(data, splits))
        return days

def concatenate_arrays(l2d, l2id):
    # get number of days
    days = 0
    for key in l2d:
        for device in l2d[key]:
            days += device.shape[0]
    data = np.ones((days, SEC_IN_DAY, 2))
    ids = []
    labels = []
    data_ind = 0
    for key in l2d:
        print('    Working on ' + key)
        for i,device in enumerate(l2d[key]):
            #print(l2id[key][i])
            np.copyto(data[data_ind:device.shape[0]+data_ind], device)
            ids += device.shape[0]*[l2id[key][i]]
            labels += device.shape[0]*[labelToNum[key]]
            data_ind += device.shape[0]
        #print(data.shape)
        #print(len(ids))
    ids = np.array(ids)
    labels = np.array(labels)
    return data, labels, ids

#read the labels file and make a dict of lists
if args.size == 'small':
    labels = smallLabels
elif args.size == 'large':
    labels = largeLabels
else:
    labelFile = open(args.labelFile,'r')
    labels = labelFile.readline().strip().split(',')

labelToFilenames = {}
labelToID = {}
labelToData = {}
labelToNum = {}
for i,label in enumerate(labels):
    labelToFilenames[label] = []
    labelToID[label] = []
    labelToData[label] = []
    labelToNum[label] = i

#now get a list of data files
dataFiles = os.listdir(args.inputdir)

for dataFile in dataFiles:
    devid = int(dataFile.split('_')[0])
    label = dataFile.split('_')[1].split('.')[0]
    if(label in labelToFilenames):
        labelToFilenames[label].append(dataFile)
        labelToID[label].append(devid)
        data = split_data(np.load(args.inputdir + '/' + dataFile))
        maximum = np.amax(data[:,:,0], 1)
        maxofmax = np.amax(maximum)
        keep = np.logical_and(maximum > (maxofmax * .1), maximum > 5)
        if label == 'Light':
            keep = np.logical_and(keep, maximum > 10)
        keptdata = data[keep]
        #print('{}, {}'.format(devid, label))
        #print(np.amax(keptdata[:,:,0], 1))
        if(data[keep].shape[0] > 0):
            labelToData[label].append(data[keep])
    else:
        print("Warning: No label for {}".format(dataFile))

print("Found following data files:")

for key in labelToFilenames:
    numFiles = len(labelToFilenames[key])
    mindays = None
    maxdays = None
    totaldays = []
    for device in labelToData[key]:
        numdays = device.shape[0]
        totaldays.append(numdays)
        if mindays is None:
            mindays = maxdays = numdays
            continue
        if mindays > numdays: mindays = numdays
        if maxdays < numdays: maxdays = numdays
    mediandays= np.median(totaldays)
    print("{}: {} files".format(key,len(totaldays)))
    totaldays= np.sum(totaldays)
    print("  number of days {}".format(totaldays))
    print("  max of days    {}".format(maxdays))
    print("  min of days    {}".format(mindays))
    print("  median of days {}".format(mediandays))
labelToUnseen = {}
labelToUnseenID = {}
labelToTrain = {}
labelToTrainID = {}

print('\nGenerate unseen set')
totalPoints = 0
for key in labelToFilenames:
    numFiles = len(labelToData[key])
    #if numFiles <= 2:
    #    print('No files for label ' + key + ', skipping...')
    #    continue
    print('Attempting to partition ' + key)

    devices = labelToData[key]
    deviceids = labelToID[key]
    if not args.house:
        # count number of points for this device
        numPoints = 0
        for data in devices:
            numPoints += data.shape[0]*data.shape[1]
            totalPoints += numPoints
        numTries = 0
        while numTries < 1000:
            pick = np.random.choice(numFiles, int(np.ceil(numFiles * unseenTestRatio)), False)
            unseenDevices = [devices[i] for i in pick]
            unseenNumPoints = 0
            for device in unseenDevices:
                unseenNumPoints += device.shape[0] * device.shape[1]
            #print(pick)
            if unseenNumPoints/numPoints > (unseenTestRatio - .1) and \
                unseenNumPoints/numPoints < (unseenTestRatio + .1):
                print(key + ' pick represents {:.2g}'.format(unseenNumPoints/numPoints))
                labelToUnseen[key] = unseenDevices
                labelToUnseenID[key] = [deviceids[i] for i in pick]
                labelToTrain[key] = [devices[i] for i in set(range(len(devices))) - set(pick)]
                labelToTrainID[key] = [deviceids[i] for i in set(range(len(devices))) - set(pick)]
                break;
            numTries += 1
        if key not in labelToUnseen:
            print('Failed to choose unseen device(s) for ' + key)
            for i,device in enumerate(devices):
                print("device {} consists of {:.2}".format(i, device.shape[0]*device.shape[1]/numPoints))
            pick = [int(i) for i in input('user pick: ').split()]
            labelToUnseen[key] = [devices[i] for i in pick]
            labelToUnseenID[key] = [deviceids[i] for i in pick]
            labelToTrain[key] = [devices[i] for i in set(range(len(devices))) - set(pick)]
            labelToTrainID[key] = [deviceids[i] for i in set(range(len(devices))) - set(pick)]
    else:
        labelToUnseen[key] = []
        labelToUnseenID[key] = []
        labelToTrain[key] = []
        labelToTrainID[key] = []
        for i in range(len(devices)):
            if deviceids[i] in houseUniqueID:
                labelToUnseen[key].append(devices[i])
                labelToUnseenID[key].append(deviceids[i])
            else:
                labelToTrain[key].append(devices[i])
                labelToTrainID[key].append(deviceids[i])
print(totalPoints)

labelToTest = {}
labelToTestID = {}
print('\nGenerate seen test set')
for key in labelToFilenames:
    numFiles = len(labelToFilenames[key])
    if numFiles <= 2:
        print('No files for label ' + key + ', skipping...')
        continue
    print('Attempting to partition ' + key)

    labelToTest[key] = []
    labelToTestID[key] = []

    devices = labelToTrain[key][:]
    labelToTrain[key] = []
    deviceids = labelToTrainID[key][:]
    labelToTrainID[key] = []

    for i in range(len(devices)):
        # pick % of device days
        if devices[i].shape[0] == 0: continue
        pick = np.random.choice(devices[i].shape[0], int(np.ceil(devices[i].shape[0] * seenTestRatio)), False)
        labelToTest[key].append(devices[i][pick])
        labelToTestID[key].append(deviceids[i])
        labelToTrain[key].append(devices[i][list(set(range(len(devices[i]))) - set(pick))])
        labelToTrainID[key].append(deviceids[i])

# concatenate sets
print('\nGenerating numpy arrays for train, test, and unseen')
print('  Working on training set')
train,trainLabels,trainID = concatenate_arrays(labelToTrain,labelToTrainID)
print('  Working on test set')
test,testLabels,testID= concatenate_arrays(labelToTest,labelToTestID)
print('  Working on unseen set')
unseen,unseenLabels,unseenID = concatenate_arrays(labelToUnseen,labelToUnseenID)

print(train.shape)
print(trainLabels.shape)
print(trainID.shape)
print(test.shape)
print(testLabels.shape)
print(testID.shape)
print(unseen.shape)
print(unseenLabels.shape)
print(unseenID.shape)
#maxs = np.max(train[:,:,0], 1)
#zeros = []
#for i,x in enumerate(maxs):
#    if x > 1: continue
#    else:
#        zeros.append(i)
#print(np.array(zeros))

np.save(args.outputdir + '/' + 'train', train)
np.save(args.outputdir + '/' + 'trainLabels', trainLabels)
np.save(args.outputdir + '/' + 'trainID', trainID)
np.save(args.outputdir + '/' + 'test', test)
np.save(args.outputdir + '/' + 'testLabels', testLabels)
np.save(args.outputdir + '/' + 'testID', testID)
np.save(args.outputdir + '/' + 'unseen', unseen)
np.save(args.outputdir + '/' + 'unseenLabels', unseenLabels)
np.save(args.outputdir + '/' + 'unseenID', unseenID)

# Need to concatenate the two sets, preserving each set's labels
# Need to split labelToTrain into training and validation
# Also need to generate class weight vector for loss function

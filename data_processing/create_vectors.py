#!/usr/bin/env python3

import numpy as np
import sys
import os
import csv
import argparse
from datetime import datetime, date
import subprocess
import glob
import matplotlib.pyplot as plt
from collections import OrderedDict

SEC_IN_DAY = 60*60*24

largeLabels = ['Blender', 'Blowdryer', 'CableBox', 'CoffeeMaker',
        'CurlingIronStraightener', 'Light',
        'Fan', 'Refrigerator', 'LaptopComputer', 'Microwave',
        'RouterModemSwitch', 'PhoneCharger', 'Television', 'Toaster']
smallLabels = ['Television', 'Refrigerator', 'Microwave', 'LaptopComputer',
        'CableBox', 'PhoneCharger', 'Toaster', 'CoffeeMaker', 'Light']

parser = argparse.ArgumentParser(description='Process data input files')
parser.add_argument('inputdir', metavar='I', type=str,
                    help='Input directory with npy array files')
parser.add_argument('labelFile', metavar='L', type=str,
                    help='A file with a comma separated list of labels')
parser.add_argument('outputdir', metavar='O', type=str,
                    help='Directory to output training and unseen npy array files')

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

data = []
for key in labelToFilenames:
    print(key)
    for i,device in enumerate(labelToData[key]):
        print('{}/{}'.format(i, len(labelToData[key])))
        for day in device[:,:,0]:
            features = OrderedDict()
            features['deviceType'] = key
            features['avgPwr'] = np.mean(day)
            features['varPwr'] = np.var(day)
            features['maxPwr'] = maxPwr = np.max(day)
            features['minPwr']= np.min(day)
            if (maxPwr > 10):
                count = np.sum(day > maxPwr*0.1)
            else:
                count = np.sum(day > maxPwr*0.5)
            features['duty'] = count/len(day)

            # calculate deltas
            dat = OrderedDict()
            dat['5'] =      [0,0]
            dat['10'] =     [0,0]
            dat['15'] =     [0,0]
            dat['25'] =     [0,0]
            dat['50'] =     [0,0]
            dat['75'] =     [0,0]
            dat['100'] =    [0,0]
            dat['150'] =    [0,0]
            dat['250'] =    [0,0]
            dat['500'] =    [0,0]

            curPow = 0
            last_real_delta = 0
            prev_delta= 0
            curSeq = 0
            totalCt = 0
            for power in day:
                if curPow == 0:
                    curPow = power
                    continue
                # Calculate delta to the previous measurement
                delta = power - curPow
                curPow = power

                # Check if this delta is combinable with the previous
                # This is true if both are valid (above 5) and have the same sign
                # If true, combine them and do not print the previous
                if abs(delta) >= 5 and abs(prev_delta) >= 5 and (delta < 0) == (prev_delta<0):
                    prev_delta = delta
                else:
                    # Otherwise, if the previous delta is valid it should be processed
                    if abs(prev_delta) >= 5:
                            # Process delta
                            # Detect potential 'spike' - positive increase followed by at least 30% decrease
                            # Note that this applies to the last real delta, as this current delta is used to confirm the 30% correction
                            if last_real_delta > 0 and (float(last_real_delta) + float(prev_delta)/.3) < 0:
                                for binSize in dat:
                                    if last_real_delta >= int(binSize)*10:
                                        dat[binSize][1] += 1
                            last_real_delta = prev_delta

                            # Assign to correct bin(s)
                            for binSize in dat:
                                if prev_delta >= int(binSize) and prev_delta <= int(binSize)*5:
                                    dat[binSize][0] += 1
                                    totalCt += 1
                    # Finally, if this current delta is valid then save it
                    if abs(delta) >= 5:
                        prev_delta = delta
                    else:
                        prev_delta = 0
            for bins in dat:
                features['ct' + bins] = dat[bins][0]
                features['spk' + bins] = dat[bins][1]

            data.append(features.copy())

print('Writing to csv at ' + args.outputdir)

f = open(args.outputdir + '/data.csv', 'w')
w = csv.DictWriter(f, data[0].keys())
w.writeheader()
w.writerows(data)
f.close()

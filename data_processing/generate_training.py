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


parser = argparse.ArgumentParser(description='Process data input files')
parser.add_argument('inputdir', metavar='I', type=str, nargs='+',
                    help='Input directory with npy array files')
parser.add_argument('labelFile', metavar='O', type=str, nargs='+',
                    help='A file with a comma separated list of labels')

args = parser.parse_args()


#read the labels file and make a dict of lists
labelFile = open(args.labelFile[0],'r')
labels = labelFile.readline().split(',')

dataFileDict = {}
for label in labels:
    dataFileDict[label] = []

#now get a list of data files
dataFiles = os.listdir(args.inputdir[0])

for dataFile in dataFiles:
    label = dataFile.split('_')[1].split('.')[0]
    if(label in dataFileDict):
        dataFileDict[label].append(dataFile)
    else:
        print "Warning: No label for {}".format(dataFile)

print dataFileDict

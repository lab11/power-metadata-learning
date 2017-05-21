#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np
import random
from subprocess import call

parser = argparse.ArgumentParser(description="Classify power data in both seen and unseen sets")

parser.add_argument('input_file')
parser.add_argument('-m')
parser.add_argument('-o')

args = parser.parse_args()

if(args.input_file[-3:] == 'csv'):
    #load it as a csv
    infile = open(args.input_file, 'r')
    field_names = infile.readline().split(',')
    field_names[-1] = field_names[-1].rstrip("\r\n")
    line = infile.readline()
    fields = line.split(',')
    for i in range(0,len(fields)):
        fields[i] = fields[i].rstrip("\r\n")

    data = [fields]
    for line in infile:
        fields = line.split(',')
        for i in range(0,len(fields)):
            fields[i] = fields[i].rstrip("\r\n")

        data.append(fields)

else:
    print("Only accepts csvs, exiting.")
    sys.exit(1)


#now add an ID field to the array and make an array that maps device to ID
id_list = range(0,len(data))
random.shuffle(id_list)
random.shuffle(id_list)
random.shuffle(id_list)
random.shuffle(id_list)


#now create a mapping of device id to the random id assigned to each day
id_map = [None]*len(data)
for i in range(0,len(data)):
    id_map[id_list[i]] = data[i][0]


#now let's take that data, write it back out to a csv for seen cross validation
cfile = open('temp_seen_in.csv','w')
cfile.write('id,')
for i in range(0,len(field_names)):
    if(i == 0):
        pass
    elif(i == len(data[0])-1):
        cfile.write(field_names[i] + '\n')
    else:
        cfile.write(field_names[i] + ',')


for i in range(0,len(data)):
    cfile.write(str(id_list[i])+',')
    for j in range(0,len(data[0])):
        if(j == 0):
            pass
        elif(j == len(data[0])-1):
            cfile.write(data[i][j] + '\n')
        else:
            cfile.write(data[i][j] + ',')

cfile.flush()
cfile.close()

#run the seen cross validation in weka and write the output to a csv
print("Generating ARFF...")
os.system('java weka.core.converters.CSVLoader temp_seen_in.csv -B 1000 > temp_seen_in.arff')
print("")
print("")
print("Running Classifier")
os.system('java weka.classifiers.trees.J48 -x 10 -t temp_seen_in.arff -classifications "weka.classifiers.evaluation.output.prediction.CSV -p first" > temp_classifier_out.csv')


#process the output csv into a numpy array
res = open('temp_classifier_out.csv','r')
res.readline()
res.readline()
res.readline()
res.readline()
res.readline()
res_data = np.zeros((len(data),4))
i = 0
for line in res:
    f = line.split(',')
    if(len(f) < 5):
        continue
    res_data[i,0] = int(f[1].split(':')[0])
    res_data[i,1] = int(f[2].split(':')[0])
    if(f[3] == ""):
        res_data[i,2] = 1
    else:
        res_data[i,2] = 0

    res_data[i,3] = int(f[5])

    i = i+1

#print day by day error
print("Day sliced error: " + str(np.mean(res_data[:,2])))

#now calculate device grouped error

















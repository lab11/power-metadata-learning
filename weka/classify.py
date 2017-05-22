

import sys
import os
import argparse
import numpy as np
import random
import subprocess

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

#also create a list of unique ids for each device
device_list = []
for i in range(0,len(data)):
    if data[i][0] not in device_list:
        device_list.append(data[i][0])

#now let's take that data, write it back out to a csv for seen cross validation
print("Generating seen test files...")
os.system('mkdir temp')
cfile = open('temp/temp_seen_in.csv','w')
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
os.system('java weka.core.converters.CSVLoader temp/temp_seen_in.csv -B 10000 > temp/temp_seen_in.arff')

#now on to unseen devices
print("Generating unseen file set...")

#first for each device in the device list create two files temp_n_unseen_train.csv and temp_n_unseen_test.csv
sys.stdout.write('\n')
proclist = set()
for i in range(0,len(device_list)):

    #find the range of instances that belong to a device
    ilist = []
    for j in range(0,len(data)):
        if(device_list[i] == data[j][0]):
            ilist.append(j+1)

    #os.system('java weka.filters.unsupervised.instance.RemoveRange -R ' + ','.join(str(e) for e in ilist) + ' < temp/temp_seen_in.arff > temp/temp_' + str(i) + '_unseen_train.arff')
    call = ['java weka.filters.unsupervised.instance.RemoveRange -R ' + ','.join(str(e) for e in ilist) + ' < temp/temp_seen_in.arff > temp/temp_' + str(i) + '_unseen_train.arff']
    p = subprocess.Popen(call, shell=True)
    proclist.add(p.pid)

    #os.system('java weka.filters.unsupervised.instance.RemoveRange -R ' + ','.join(str(e) for e in ilist) + ' -V < temp/temp_seen_in.arff  > temp/temp_' + str(i) + '_unseen_test.arff')
    call = ['java weka.filters.unsupervised.instance.RemoveRange -R ' + ','.join(str(e) for e in ilist) + ' -V < temp/temp_seen_in.arff  > temp/temp_' + str(i) + '_unseen_test.arff']
    p = subprocess.Popen(call, shell=True)
    proclist.add(p.pid)

    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(str(i))
    sys.stdout.flush()

while proclist:
     try:
         pid,retval = os.wait()
     except:
         #just assume this means were done
         break

     proclist.remove(pid)



print("")
print("")
print("Running Seen Classifier...")
os.system('java weka.classifiers.trees.J48 -x 10 -t temp/temp_seen_in.arff -classifications "weka.classifiers.evaluation.output.prediction.CSV -p first" > temp/temp_classifier_out.csv')


#process the output csv into a numpy array
res = open('temp/temp_classifier_out.csv','r')
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


#now calculate device grouped error
#first construct an array of the correct label for each device
correct_labels = np.zeros(len(device_list))
for i in range(0,len(correct_labels)):
    first_id_ind = id_map.index(device_list[i])
    res_ind = np.where(res_data[:,3] == first_id_ind)
    correct_labels[i] = res_data[res_ind,0]


#iterate through and generate a vote map
votes = np.zeros((len(device_list),int(max(res_data[:,1])+1)))
for i in range(0,len(res_data)):
    uid = res_data[i,3]
    vote = res_data[i,1]
    devid = id_map[int(uid)]
    devind = device_list.index(devid)
    votes[devind,int(vote)] += 1

#take the argmax for the votes
fin_vote = np.argmax(votes,1)
dev_grouped_correct = (fin_vote == correct_labels)

print("Running Unseen Classifier...")
total_devices = 0
correct_devices = 0
sys.stdout.write('\n')
#proclist = set()
for i in range(0,len(device_list)):
    os.system('java weka.classifiers.trees.J48 -t temp/temp_' + str(i) + '_unseen_train.arff \
                -T temp/temp_' + str(i) + '_unseen_test.arff -classifications \
                "weka.classifiers.evaluation.output.prediction.CSV -p first" > temp/temp_' + str(i) + '_unseen_out.csv')
    #call = ['java weka.classifiers.trees.RandomForest -t temp/temp_' + str(i) + '_unseen_train.arff \
    #            -T temp/temp_' + str(i) + '_unseen_test.arff -classifications \
    #            "weka.classifiers.evaluation.output.prediction.CSV -p first" > temp/temp_' + str(i) + '_unseen_out.csv']

    #p = subprocess.Popen(call, shell=True)
    #proclist.add(p.pid)
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(str(i))
    sys.stdout.flush()


#while proclist:
#     try:
#         pid,retval = os.wait()
#     except:
#         #just assume this means were done
#         break
#
#     proclist.remove(pid)
#
for i in range(0,len(device_list)):
    #process the output csv into a numpy array
    res2 = open('temp/temp_' + str(i) + '_unseen_out.csv','r')
    res2.readline()
    res2.readline()
    res2.readline()
    res2.readline()
    res2.readline()
    actual = 0
    dev_votes = np.zeros(int(max(res_data[:,1])+1))
    for line in res2:
        f = line.split(',')
        if(len(f) < 5):
            continue

        dev_votes[int(f[2].split(':')[0])] += 1
        actual = int(f[1].split(':')[0])

    if(np.argmax(dev_votes) == actual):
        correct_devices += 1
        print("correct")
    else:
        print("wrong")

    total_devices += 1

print("Day sliced accuracy: " + str(np.mean(res_data[:,2])))
print("Device grouped accuracy: " + str(np.mean(dev_grouped_correct)))
print("Unseen device accuracy: " + str(float(correct_devices)/total_devices))


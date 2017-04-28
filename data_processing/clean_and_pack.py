#the goal of this file is to take a tab separated output file from the SQL database
#then store it in a numpy array.

#it should be cleaned such that data appears every second

#it will be stored such that data is 24 hour aligned

import numpy as np
import sys
import argparse
from datetime import datetime, date


parser = argparse.ArgumentParser(description='Process data input files')
parser.add_argument('inputfile', metavar='I', type=str, nargs='+',
                    help='input file to parse')
parser.add_argument('outputfile', metavar='O', type=str, nargs='+',
                    help='output file to save')

args = parser.parse_args()

#open the input file
print args.inputfile
infile = open(args.inputfile[0],'r')

#figure out how long the input file is to create a numpy array for it
infile.readline()
time1 = infile.readline().split('\t')[1]
print time1

infile.seek(-100,2)
line = infile.readlines()
time2 = line[-1].split('\t')[1]
print time2

#no calculate the number of days of data we need
time_start = datetime.strptime(time1,"%Y-%m-%d %H:%M:%S")
time_end = datetime.strptime(time2,"%Y-%m-%d %H:%M:%S")

end_start_day = None
#round the first day up to midnight
if(time_start.hour == 0 and time_start.minute == 0):
	pass
	end_start_day = time_start
else:
	day_time = time_start.day
	end_start_day = time_start.replace(day=day_time+1,hour=0,minute=0,second=0) 

#round the end time down to
end_end_day = None
if(time_end.hour == 23 and time_end.minute == 59 and time_end.second >50):
	end_end_day = time_end.replace(second=59)
else:
	day_time = time_end.day
	end_end_day= time_end.replace(day=day_time-1,hour=23,minute=59,second=59) 

#now subtract the two days
difference = end_end_day - end_start_day
size_of_array = difference.total_seconds() + 1

#make a numpy array
array_to_store = np.zeros((int(size_of_array),2))

#seek back to beginning of file
infile.seek(0)
infile.readline()

#find the first date that matches the start date
last_time = None
it = 0
for line in infile:
	seq,time,power,pf = line.split('\t')
	time = datetime.strptime(time,"%Y-%m-%d %H:%M:%S")
	if(time >= end_start_day):
		last_time = time
		array_to_store[it,0] = float(power)
		array_to_store[it,1] = float(pf)
		it = it+1
		break


for line in infile:
	seq,time,power,pf = line.split('\t')
	time = datetime.strptime(time,"%Y-%m-%d %H:%M:%S")
	diff = time-last_time
	if(diff.total_seconds() == 1):
		#everything is fine
		array_to_store[int(it),0] = float(power)
		array_to_store[int(it),1] = float(pf)
		it = it+1
	elif(diff.total_seconds() < 1):
		#just ignore this point
		pass
	else:
		sec = diff.total_seconds()
		for i in range(0,int(sec)):
			newit = it+i
			array_to_store[int(newit),0] = float(power)
			array_to_store[int(newit),1] = float(pf)
		it = it+sec

	last_time = time

np.save(args.outputfile[0],array_to_store)	













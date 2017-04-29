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
infile = open(args.inputfile[0],'r')
outfile = open(args.outputfile[0],'w')
outfile.write(infile.readline())

last_line = infile.readline()
for line in infile:
        last_seq,last_time,p,pf = last_line.split('\t')
	seq,time,power,pf = line.split('\t')

        if(time == last_time):
            if(seq < last_seq):
                outfile.write(line)
            else:
                outfile.write(last_line)
                last_line = line
        else:
            outfile.write(last_line)
            last_line = line

outfile.write(last_line)

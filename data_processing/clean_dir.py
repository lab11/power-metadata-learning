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
                    help='input file to parse')
parser.add_argument('outputdir', metavar='O', type=str, nargs='+',
                    help='output file to save')

args = parser.parse_args()

if(os.path.abspath(args.inputdir[0]) == os.path.abspath(args.outputdir[0])):
    print "Cannot put files into same directory! Exiting..."
    sys.exit(1)

#get a list of all files in input directory
inFileList = os.listdir(args.inputdir[0])

proclist = set()
for file in inFileList:
    outFileName = file[:-3]+"npy"
    id = outFileName.split('_')[0]
   
    call = ['python','clean_file.py',args.inputdir[0]+file,args.outputdir[0]+outFileName,id]
    p = subprocess.Popen(call)
    proclist.add(p.pid)
    
while proclist:
    try:
        pid,retval = os.wait()
    except:
        #just assume this means were done
        break

    proclist.remove(pid)

print "Consolidating reports into single file.."

reportList = glob.glob(args.outputdir[0] + "*.rpt")
creport = open("cleaning_report.rpt",'w')
creport.write("ID,days,percent_interpolated,percent_off,percent_skipped\n")

for r in reportList:
    rep = open(r,'r')
    line = rep.readline()
    creport.write(line)
    rep.close()
    os.remove(r)

print "Finished all cleaning process!"

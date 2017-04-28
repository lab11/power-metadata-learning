import sys
import os
from datetime import datetime, date, timedelta
from subprocess import call

DATA_DIR = "../data/"
DEVICE_FILE = "../devices/devices_label_anon.csv"
PASSWORD_FILE = "password.txt"

passf = open(PASSWORD_FILE,'r')
uname = passf.readline()[:-1]
password = passf.readline()[:-1]

print "Successfully parsed password file!"

#first let's list the files in the data_dir and see what we've done so far
retrieved = os.listdir(DATA_DIR)

#parse the filenames
#id_label_startdate_enddate.tab
id_retrieved = 0
for fileName in retrieved:
    splits = fileName.split('_');
    id = int(splits[0])
    id_retrieved = id


id_to_retrieve = id_retrieved+1

if(id_retrieved == 0):
    print "Found no data already retrieved - starting from scratch"
elif(id_retrieved == 1):
    print "Found data for ID 1".format(id_retrieved)
else:
    print "Found data for IDs 1-{}".format(id_retrieved)

#now open the devices file
device_info = open(DEVICE_FILE,'r')
device_info.readline()
for device in device_info:
    id, mac, type, category, room, start, end, label = device.split(',')
    label = label[:-2]

    if(int(id) < id_to_retrieve):
        continue
    
    #okay now we need to retrieve the id
    print "Analyzing data for device {} with label {} ...".format(id,label)

    #How much data do we have?
    startTime = datetime.strptime(start,"%Y-%m-%d %H:%M:%S")
    endTime = datetime.strptime(end,"%Y-%m-%d %H:%M:%S")

    diff = endTime - startTime
    print "Device has {} days of data".format(diff.days)

    if(diff.days > 60):
        print "Too much data - truncating to 60 days"
        endTime = startTime + timedelta(days=60)

    end = endTime.strftime("%Y-%m-%d %H:%M:%S")

    print "Starting to retrieve ID {} from {} to {}. This may take a few minutes".format(id,start,end)
    print "."
    print "."
    print "."
    print "."
    estring =   """"SELECT seq,timestamp,power,pf """ +  \
                """FROM dat_powerblade """  + \
                """ WHERE deviceMAC= '""" + mac + """' """ + \
                """ AND timestamp > '""" + start + \
                """' AND timestamp < '""" + end +  \
                """' order by timestamp asc" """;

    hostname = "ins-pb-deploy.cgy7hjsdf4rc.us-west-2.rds.amazonaws.com"
    outfileName = DATA_DIR + "{}_{}.tab".format(id,label)
    outfile = open(outfileName,'w')
    calling = 'mysql -h ' + hostname + ' --user=' + uname + ' --password=' + password + ' --database powerblade -B -e ' + estring
    call(calling,shell=True,stdout=outfile)
    
    #now that we have returned from the call let's start a background process to clean it
    #infileName = outfileName
    #outfileName = DATA_DIR + "{}_{}.npy".format(id,label)
    #calling = 'python clean_and_pack.py ' +  infileName + ' ' + outfileName + ' &'
    #print infileName
    #print outFileName
    #print calling
    #call(calling,shell=True)

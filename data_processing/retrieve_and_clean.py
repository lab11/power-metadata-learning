import sys
import os

DATA_DIR = "../data/"
DEVICE_FILE = "../devices/devices_label_anon.csv"

#first let's list the files in the data_dir and see what we've done so far
retrieved = os.listdir(DATA_DIR)

#parse the filenames

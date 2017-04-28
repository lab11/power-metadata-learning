Retrieving, Cleaning and Packing
===============================

To facilitate fast processing of the data we are pulling relevant data
from the mysql database and storing it in binary files

We will also do some data cleaning

- clear_and_pack.py takes an input file, cleans (interpolates to get data every
second), and packs the data into a numpy file starting at the soonest midnight
and going until the last midnight before the end of the input data. The
script expects a tab separated file of sequence_number, timestamp, power, power factor.

- retrieve_and_clean.py uses the devices list under devices to pull data
from the database, then runs the clean_and_pack script on the retrieved data.

To use the retrieve script, you must place a password.txt file in this
directory with credentials to access the database.

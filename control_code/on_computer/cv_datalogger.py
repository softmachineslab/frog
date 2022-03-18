#!/usr/bin/python3

# This data logger adapted from brittlestar-ros to save april tag / states as well as command outputs.

# packages to communicate with the OS and save files
import sys
from datetime import date
#from datetime import time
from datetime import datetime
import os
import numpy as np

# for calling outside of this class - TO-DO FIX THIS HACK
def get_time_since_midnight_singleton():
    # get the current time (clock ticks of computer)
    now = datetime.now()
    # subtract from midnight. Midnight is
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # total microseconds, according to system clock, since midnight.
    # Is a float.
    usec = (now - midnight).total_seconds()
    # Turn into a in, with microseconds only.
    msec = int(round(usec * 1000))
    return msec

# both the class and others will need to expand out a base directory to ymdh.
def get_ymdh_folder_path():
    # Returns:
    # year/year_month/year_month_day/year_month_day_hour
    t = date.today()
    ymdh_path = str(t.year)
    ymdh_path = os.path.join(ymdh_path, str(t.year) + "_" + str(t.month))
    ymdh_path = os.path.join(ymdh_path, str(t.year) + "_" + str(t.month) + "_" + str(t.day))
    ymdh_path = os.path.join(ymdh_path, str(t.year) + "_" + str(t.month) + "_" + str(t.day) + "_" + str(datetime.now().hour))
    return ymdh_path

class CVDataLogger:

    # Some functions to calculate timestamps/
    # First, milliseconds since midnight.
    # This makes it easiest to correlate timestamps when post-processing,
    # Rounded to an int.
    # Thanks to https://stackoverflow.com/questions/15971308/get-seconds-since-midnight-in-python
    def get_time_since_midnight(self):
        # get the current time (clock ticks of computer)
        now = datetime.now()
        # subtract from midnight. Midnight is
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # total microseconds, according to system clock, since midnight.
        # Is a float.
        usec = (now - midnight).total_seconds()
        # Turn into a in, with microseconds only.
        msec = int(round(usec * 1000))
        return msec

    # This one returns a string of the date and starting time, for use
    # with file naming.
    def get_datetime_string(self):
        # Should be of the form
        # Year-Month-Day_HourMinSec
        t = date.today()
        stampdate = str(t.year) + "-" + str(t.month) + "-" + str(t.day)
        # Now, for the time itself
        now = datetime.now()
        stamptime = now.strftime("%H%M%S")
        # the full datetime timestamp is
        stamp = stampdate + "_" + stamptime
        return stamp

    # The function that actually logs the data.
    def log_callback(self, data):
        # Open the file, append mode
        f = open(self.filename, 'a')
        # Construct a comma-separated row to save.
        # Assume that 'data' is a list of objects that can be converted to strings via str()
        row = str(self.get_time_since_midnight())
        for element in data:
            row = row + "," + str(element)
        # finish with a newline
        row = row + "\n"
        # write the file. It may have been easier to use np.savetxt but whatever
        f.write(row)
        f.close()

    # a helper function for startup, called by constructor
    def datalogger_startup(self, file_folder, curve):
        # add a folder tree for better organization
        file_folder = os.path.join(file_folder)
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        # Create the filename. It's a concatenation of the folder with a descriptive name,
        # and a timestamp.
        # filename = file_folder + "/cv_datalogger_" + self.get_datetime_string() + ".csv"
        # Indpendent of operating system:
        filename_local = "cv_datalogger_" + self.get_datetime_string() + ".csv"
        filename_path_local = "cv_datalogger_path_" + self.get_datetime_string() + ".csv"
        filename = os.path.join(file_folder, filename_local)
        filename_path = os.path.join(file_folder, filename_path_local)
        print("Running cv_datalogger. Writing to file " + filename)
        # Open the file and put a header.
        f = open(filename, 'a')
        f.write("CV Data Logger started on:," + self.get_datetime_string())
        f.write("\n")
        np.savetxt(filename_path, curve, delimiter=',')
        # To-do: have someone else pass in the header.
        # f.write("Timestamp (millisec since midnight today),Test time,PWM1 value,PWM2 value,x_0,y_0,x_1,y_1,...\n")
        f.write("Timestamp (millisec since midnight today),x,y,theta,xdot,ydot,thetadot,cost,last action\n")
        f.close()
        self.filename = filename

    # the constructor initializes everything.
    # Here is where the file handle is created.
    def __init__(self):
        # Save a reference to the file name so the callback can append new data.
        pass
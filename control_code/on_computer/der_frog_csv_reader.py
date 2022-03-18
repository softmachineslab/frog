# Read from CSV files produced by the DER simulations.
# Designed for the frog simulations.
# Andrew Sabelhaus / SML 2020-06-25

# from environment
# need regexps for parsing hour strings
import re 
# get the file separator for this operating system
import os
# numpy for reading arrays from csv files
import numpy as np

def load_der_frog_csv_dataset(first_hr, last_hr, logfile_base, start_row=3):
    # Correct for initial tilde if passed in.
    logfile_base = os.path.expanduser(logfile_base)
    # alternatively, if the path is relative, convert to absolute. easier debugging.
    logfile_base = os.path.abspath(logfile_base)
    # List all the folder paths where we should load data from.
    all_paths = []
    # Our regular expression for the year, month, day, hr string is just matching an underscore
    ymdh_pat = re.compile('_')
    # get the year, month, day, hour from each and convert into integers.
    first_ymdh_str = ymdh_pat.split(first_hr)
    last_ymdh_str = ymdh_pat.split(last_hr)
    first_ymdh = list(map(int, first_ymdh_str))
    last_ymdh = list(map(int, last_ymdh_str))

    # Count and build up the list of file paths.
    # TO-DO: make this neater. Hard to do >= with dates...
    paths_finished = False
    # separate counter
    next_ymdh = first_ymdh
    while not paths_finished:
        # Add in the next path
        all_paths.append(get_ymdh_logfile_path(logfile_base, next_ymdh))
        # check if we're done
        if next_ymdh == last_ymdh:
            paths_finished = True
        else:
            # increment
            next_ymdh = increment_ymdh(next_ymdh)
    # should have all paths now.
    
    # For ease, do a list of numpy arrays. We'll transform into a better data structure while parsing later.
    der_csv_data_list = []
    # Next, for each directory, get all files in it, and read from each file.
    for pth in all_paths:
        print('Reading from directory ' + pth + ' ...')
        # list of all files. Thanks to stack overflow, https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
        all_files_i = [f for f in os.listdir(pth) if os.path.isfile(os.path.join(pth, f))]
        # Helper will give us the data from this file.
        for f in all_files_i:
            # need to concatenate with directory again
            f = os.path.join(pth, f)
            all_d = load_der_csv_onefile(f, start_row)
            # concatenate to our result
            der_csv_data_list.append(all_d)
    
    # return the list for now. It's implied that each file will represent one transition.
    return der_csv_data_list

# helper: read the CSV data from a given file.
def load_der_csv_onefile(csv_path, start_row):
    # starts at
    all_data = np.genfromtxt(csv_path, delimiter=',', skip_header=(start_row))
    # here: process if needed.
    return all_data

# helper: increment the year month day hour.
def increment_ymdh(ymdh):
    # Increment counters, with overflow. Assumes ymdh is a list of 4 integers.
    # hours increment
    ymdh[3] += 1
    if ymdh[3] > 23:
        # midnight
        ymdh[3] = 0
        # days increment
        ymdh[2] += 1
        if ymdh[2] > 31:
            # NOTE: FIX THIS, ERRORS ON NOT-31-DAY MONTHS
            # first day of month
            ymdh[2] = 1
            # month increment
            ymdh[1] += 1
            if ymdh[1] > 12:
                # january
                ymdh[1] = 1
                # year increment
                ymdh[0] += 1
    return ymdh

# helper: concatenate strings together for the logfile path.
def get_ymdh_logfile_path(logfile_base, ymdh):
    # Add the path string for this year, month, day, hour
    # that path string would be (for example)
    # 2020/2020_5/2020_5_7/2020_5_7_17/
    # Assumes ymdh is a list of 4 integers.
    return os.path.join(logfile_base, str(ymdh[0]), \
        str(ymdh[0]) + '_' + str(ymdh[1]), \
        str(ymdh[0]) + '_' + str(ymdh[1]) + '_' + str(ymdh[2]), \
        str(ymdh[0]) + '_' + str(ymdh[1]) + '_' + str(ymdh[2]) + '_' + str(ymdh[3]))
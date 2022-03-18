# Example script that reads in the DER frog simulation data and would create the data structure for a transition model
# Andrew Sabelhaus / SML 2020-06-25

# from the environment
import numpy as np
# our modules
import der_frog_csv_reader

print('Loading DER simulation dataset for the frog robot...')

# Parameters for import: 
# path to the directory where we're storing the DER simulation data for the frog.
# Depending on what directory you run this from...
logfile_base = 'data/der_sim_data/frog/'
# timestamps to read from. If first = last, we're only looking at one hour.
first_hr = '2020_6_25_13'
last_hr = first_hr

# Import. At this point, just the raw data, the form of a list.
der_frog_data_list = der_frog_csv_reader.load_der_frog_csv_dataset(first_hr, last_hr, logfile_base)

print('Received ' + str(len(der_frog_data_list)) + ' transition observations.')
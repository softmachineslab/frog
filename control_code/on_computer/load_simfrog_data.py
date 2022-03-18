#!/usr/bin/python3

# Demonstration of loading the simFrogDER data and organizing into action sets. 

import numpy as np
import pickle
import os
import csv
import pandas as pd

print("simFrogDER data importer.")

# Notes from the C++ simulator: bounds on sampling for x_dot, y_dot, omega are: (meter/sec and degrees/sec)
# vector<double> lower_unif_bnd = {-0.03, 0, -30};
# vector<double> upper_unif_bnd = {0.03, 0.04, 30};

##########################################################
# Paths for every data file
logfile_root = '../log_files_2021-11-13/log_files'
allfilenames = []
for dirpath, dirnames, filenames in os.walk(logfile_root):
    fullfnames = [os.path.join(dirpath, f) for f in filenames]
    allfilenames.extend(fullfnames)
print("There are " + str(len(allfilenames)) + " log files. Starting to read in... this may take a while...")

##########################################################
# Setup
tf = 3.0 # seconds, final time at which action is "done" i.e. control frequency

# num_actions = 9 # we know this ahead of time, can be used to confirm we read in our data correctly

# Reload the data versus use a pickled version already
reload_data = False
raw_sims_ungrouped_file = 'simfrog_rawsims_2021-11-29_3pm.pickle'

##########################################################
# IF RELOADING: Parse all into a really inefficient data structure just to start

if reload_data:
    sims = []
    loaditer = 0
    totalloaditer = len(allfilenames)
    for f in allfilenames:
        print(str(loaditer) + "/" + str(totalloaditer))
        loaditer = loaditer+1
        # sometimes the simulation crashes or stops before we reach our desired time.
        simcrashed = False
        # get the action and starttime for this file (will let us organize actions-per-initial-condition)
        with open(f, 'r') as openf:
            rdr = csv.reader(openf, delimiter=',')
            rows = list(rdr)
            action = rows[1][1]
            timestamp = rows[2][1]
            openf.close()
        # the simulation results
        fdata = np.genfromtxt(fname=f, delimiter=',', skip_header=4)
        # first column is time. Remainder is rigid body state,
        # \mathbf{x} = [x, y, theta, dxdt, dydt, omega]
        #       units: [m, m, deg,   m/s,  m/s,  deg/s]
        # if the simulation crashes immediately don't even examine it - that's one row or less.
        if fdata.ndim > 1:
            x0 = fdata[0, 1:]
            tf_idx = np.argmax(fdata[:,0] >= tf)
            xf = fdata[tf_idx, 1:]
            if fdata[tf_idx,0] != tf:
                simcrashed = True
            if not simcrashed:
                sim_i = {"initcond_timestamp":timestamp, "action":action, "x0":x0, "xf":xf}
                sims.append(sim_i)
    print("There were " + str(len(sims)) + " valid simulations imported.")
    # Pickle the result so we don't have to do this a million times
    pfilenameraw = 'simfrog_rawsims_2021-11-29_3pm.pickle'
    with open(pfilenameraw, 'wb') as pfile:
        pickle.dump(sims, pfile)
else:
    print('Loading raw simulations from a parsed pickle file...')
    with open(raw_sims_ungrouped_file, 'rb') as pfile:
        sims = pickle.load(pfile)

##########################################################
# Reorganize.
# This is where the initial condition timestamp comes in handy. 
# It *should* be the case that there are num_actions-many simulations with the same initial condition i.e. the same initial condition timestamp.

print("Now grouping simulations together by initial condition...")
simiter = 0
totalsimiter = len(sims)

grouped_sims = {}
for s in sims:
    print(str(simiter) + "/" + str(totalsimiter))
    simiter = simiter+1
    current_initcond_sims = {}
    # check and replace if we've already seen a few simulations from this initial condition
    s_t = s["initcond_timestamp"]
    if s_t in grouped_sims.keys():
        current_initcond_sims = grouped_sims[s_t]
        # print('Found timestamp: ' + str(s_t))
        # TODO confirm that this simulation is indeed at the same initial condition as its friends already in this set
    # add this simulation to the action set for this initial condition. kind lazy here but you get the idea
    current_initcond_sims[s["action"]] = s
    grouped_sims[s_t] = current_initcond_sims

##########################################################
# Now we can do something like...
# here's an arbitrary example
example_initcond_simset = next(iter(grouped_sims.values()))
example_initcond_timestamp = next(iter(grouped_sims.keys()))

print("Example: the first set of simulations had an initial condition sampled at time ")
print(str(example_initcond_timestamp))
print("The initial condition was ")
# grab from one of the actions - these SHOULD all be the same, see the TODO above
example_fromthisset = next(iter(example_initcond_simset.values()))
example_initcond_fromthisset = example_fromthisset["x0"]
print(str(example_initcond_fromthisset))
print("The available actions and the transition results are ")
for action_name, transition in example_initcond_simset.items():
    print("Action: " + action_name)
    print("Result: " + str(transition["xf"]))
glist = list(grouped_sims.values())
print(example_initcond_simset)
##########################################################
# Pickle THIS result too
# pickle.dump(grouped_sims, open(pfilename, "wb"))
pfilenamegrouped = 'simfrog_groupedsims_2022-01-26_1pm.pickle'
with open(pfilenamegrouped, 'wb') as pfile:
    pickle.dump(grouped_sims, pfile)

dummy = [0.0, 0.0, 0.0, 0.1, 0.1, 5]
tstamp_column = {'Timestamp': list(grouped_sims.keys())}
df = pd.DataFrame()
# df['x0'] = np.nan
# df['y0'] = np.nan
# df['t0'] = np.nan
# df['xdot0'] = np.nan
# df['ydot0'] = np.nan
# df['tdot0'] = np.nan
# df['a'] = np.nan
# df['G'] = np.nan
# df['H'] = np.nan
# df['I'] = np.nan
keys = ['a','x0','y0','t0','xdot0','ydot0','tdot0','xf','yf','tf','xdotf','ydotf','tdotf']
simiter = 0
data = np.empty((0,12))
numpy_actions = []
for initcond_timestamp in iter(grouped_sims.keys()):
    
    initcond_simset = grouped_sims[initcond_timestamp]
    init_cond_dict = next(iter(initcond_simset.values()))
    init_cond = init_cond_dict["x0"]
    for action_name, transition in initcond_simset.items():
        print(str(simiter) + "/" + str(totalsimiter))
        simiter = simiter+1
        # print("Action: " + action_name)
        # print("Result: " + str(transition["xf"]))
        row_values = [action_name[18]] + list(init_cond) + list(transition["xf"])
        data = np.append(data, np.array([list(init_cond) + list(transition["xf"])]), axis=0)
        numpy_actions.append(action_name[18])
        #row_dict = dict(zip(keys,row_values))
        #df = df.append(pd.DataFrame([row_dict]), ignore_index = True)
    
#print(df)
np.save('simfrog_numpy_data_2022-01-26.npy',data)
np.save('simfrog_numpy_actions_2022-01-26.npy',numpy_actions)
print(np.shape(data))
#df.to_pickle('simfrog_df_2021-11-30.pickle')
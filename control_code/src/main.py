"""
    Code for running planning and control software using DER library for soft robot path following. 
    Paper under review
    Copyright Soft Machines Lab 2022
    (license TBD)
"""

from importlib.resources import path
import sys

from multiprocessing import Process, Queue, Pipe

import april_tracking as vision

import planner as control

import cv2

import time

import serial

import multi_serial_comms as m_serial

import cv_datalogger

import os
from datetime import date
from datetime import datetime

from queue import Empty

import numpy as np

# TO DO
# Test Discounting and Atkeson Algo
# Specify paths based on pixels instead of in world space


def start_robot(device_name, save_directory, calibration = False, mode = 'brute', shape = 'li', depth = 3, model='simfrog_numpy_data_2022-01-26.npy'):
    

    video_save_directory = save_directory
    datalogger = cv_datalogger.CVDataLogger()
    # add a folder tree for better organization
    video_save_directory = os.path.join(video_save_directory, cv_datalogger.get_ymdh_folder_path())
    if not os.path.exists(video_save_directory):
        os.makedirs(video_save_directory)
    # os-independent:
    video_savefile = os.path.join(video_save_directory, "frog_video_tracking_" + datalogger.get_datetime_string())
    
    

    state_queue = Queue()
    path_queue = Queue()
    tracker = vision.Tracker(state_queue,path_queue,video_savefile,False,calibration)
    cam_process = Process(target=tracker.cv_process, args=())
    cam_process.start()

    sender, receiver = Pipe()
    serial_process = Process(target=m_serial.echo_to_terminal, args=(device_name,sender))
    serial_process.start()

    serial_port = serial.Serial(port=device_name, baudrate=115200, timeout =1,
                                        exclusive=False)

    
    time.sleep(0.5)
    state = state_queue.get()
    #state = [234.577425645519, -324.91754761909897, 2.862985398826103, 46696.20477905567, -64679.78023962873, 569.9207930820085]
    #state = [243.58021183148907, -292.1691060590805, 2.7840179000933003, 3.366293857426664, 6.922470257854112, -0.11067967126431041]
    print(state)
    width = 0.400
    height = 0.50
    conversion, center = path_queue.get()
    path = control.Bezier(conversion, center, width, height, state, 100, shape)

    path_queue.put(path.get_curve())
    path.plot()
    frog = control.Robot(model,state[0],state[1],state[2],state[3],state[4],state[5],state_queue, mode)

    planner = control.FrogPlanner(frog, path, serial_port,depth)

    planner_process = Process(target=planner.plan)
    planner_process.start()

    activation_time = 80
    period = 1.5

    last_action_queue = Queue()
    executor_process = Process(target=control.executor, args=(planner,activation_time,period,last_action_queue))
    executor_process.start()

    data_savefile = os.path.join(video_save_directory, "frog_data_tracking_" + datalogger.get_datetime_string())
    curve = path.get_curve()
    datalogger.datalogger_startup(data_savefile,curve)

    last_action = 'I'
    while True:

        
        # get frame from the queue
        state = state_queue.get()
        while not last_action_queue.empty():
            try:
                last_action = last_action_queue.get_nowait()
            except Empty:
                last_action = last_action
        total_cost, _ = path.cost(np.array(state).reshape(1,6))
        state.append(total_cost)
        state.extend(costs)
        state.append(last_action)
        datalogger.log_callback(state)
        #time.sleep(0.1)
        # cognition.dummy(serial_port)
        #receiver.recv()

if __name__ == '__main__':
    try:
        # to run the program, call main from a terminal with the necessary arguments
        # argv[1] T/F if there is a calibration grid. The code works well without it, so I use 'False'
        # argv[2] Planner uses nearest neighbor ('brute') or locally weighted regression ('lwr') to determine the next state
        # argv[3] Type of path to follow. Options are line 'li', sinusoid 'si', and ellipsoid 'el'
        # argv[4] Depth of tree of the planner (integer). Works well with smaller values (e.g. 2)
        # argv[5] is the numpy file containing the transition models from the DER library (e.g. 'simfrog_numpy_data_2022-01-26.npy')
        # argv[6] is the save directory for video and data
        # example:
        # python3 main.py 'False' 'brute' 'si' 1 'simfrog_numpy_data_2022-01-26.npy' 'data/my_save_directory'
        start_robot('/dev/ttyACM0', sys.argv[6], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        
    except KeyboardInterrupt:
        # why is this here?
        pass



from importlib.resources import path
import sys

from multiprocessing import Process, Queue, Pipe

import april_tracking as vision

import new_planner as control

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


def start_robot(device_name, calibration = False, mode = 'brute', shape = 'li', depth = 3):
    

    video_save_directory = "/media/Zach/Shared/MEGAsync/SML/sea_star/Frog"
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
    frog = control.Robot(state[0],state[1],state[2],state[3],state[4],state[5],state_queue, mode)

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
        total_cost, costs = path.cost(np.array(state).reshape(1,6))
        state.append(total_cost)
        state.extend(costs)
        state.append(last_action)
        datalogger.log_callback(state)
        #time.sleep(0.1)
        # cognition.dummy(serial_port)
        #receiver.recv()

if __name__ == '__main__':
    try:
        # the 0-th arg is the name of the file itself, so we want the 1st.
        start_robot('/dev/ttyACM0', False, 'brute', 'el', 1)
    except KeyboardInterrupt:
        # why is this here?
        pass



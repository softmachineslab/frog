#!/home/zpatty/anaconda3/envs/cython-dev/bin/python

# import async libraries
import asyncio
from asyncio import get_event_loop

# async pyserial
from serial_asyncio import open_serial_connection

# library for async equivalent of input()
from aioconsole import ainput

import sys

import csv

import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd







async def run_connection(action_generator, device_name):
    # open serial connection through the Streams API and get reader and writer objects
    reader, writer = await open_serial_connection(url=device_name, baudrate=115200)

    # relay status to terminal
    print("Opened port. Press reset on Central to initialize echoing. Ctrl-C to stop.")

    # execute reader and writer coroutines concurrently
    await asyncio.gather(read(reader),write(writer, action_generator))

# Reads from serial, converts all received messages to strings, and prints to terminal
async def read(reader):
    filename = f"strain-output.csv"
    plt.ion()
    plt.style.use('seaborn')
    plt.style.use('fast')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.show()
    time_list = []
    strain1_list = []
    while True:
        line = await reader.readuntil(b'\n')
        line = str(line, 'utf-8')
        #if line[9:12] == "Str":
        if line[0:3] == "Str":
            print(line)
            meas1 = line[9:9+6]
            print(meas1)
            print(twos_comp(int(meas1,16), 24)/(2**19))
            meas2 = line[28:28+6]
            print(meas2)
            print(twos_comp(int(meas2,16), 24)/(2**19))


            scale = 16 ## equals to hexadecimal

            num_of_bits = 32
            #print(bin(int(meas1, scale)))
            #print(int(meas1, scale)/(2**19))

            now_time = time.time()
            out1 = twos_comp(int(meas1,16), 24)/(2**19)
            out2 = twos_comp(int(meas2,16), 24)/(2**19)
            #out2 = twos_comp(int(meas2,16), 32)
            #print('Out 1: ' + str(out1))
            #time_list = time_list.append(now_time)
            #strain1_list = strain1_list.append(out1)
            
            ax.scatter(now_time, out1, c = "red")
            #ax.scatter(now_time, out2, c = "blue")
            plt.pause(0.000000000000001)
            with open(filename,"a") as f:
                writer = csv.writer(f,delimiter=",")
                writer.writerow([out1/(2**19)])
            #print('Out 2: ' + str(out2/(2**19)))

        else:
            print(line)

# Listen to terminal for user input and relay messages to serial
async def write(writer, action_generator):

    # this wait is not functional, it is just to give the user time to reset Central
    await asyncio.sleep(6) 

    while True:
        
        msg = await action_generator.get_action() 
        msg = msg + '\n'
        #print(msg)
        print('Sending')
        writer.write(msg.encode('UTF-8'))

def serial_process_start(action_generator, device_name):
    asyncio.run(run_connection(action_generator, device_name))

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val      

def twos(val_str, bytes):
    import sys
    val = int(val_str, 2)
    b = val.to_bytes(bytes, byteorder=sys.byteorder, signed=False)                                                          
    return int.from_bytes(b, byteorder=sys.byteorder, signed=True)

class CmdLnActionGenerator:
        def __init__(self):
            pass
        def get_action(self):
            return ainput("Message to send over serial terminal: ")

if __name__ == '__main__':
    
    serial_process_start(CmdLnActionGenerator(), sys.argv[1])
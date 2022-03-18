#!/home/zpatty/anaconda3/envs/cython-dev/bin/python

# import async libraries
import asyncio
from asyncio import get_event_loop

# async pyserial
from serial_asyncio import open_serial_connection

# library for async equivalent of input()
from aioconsole import ainput


async def run_connection(action_generator):
    # open serial connection through the Streams API and get reader and writer objects
    reader, writer = await open_serial_connection(url='/dev/ttyACM0', baudrate=115200)

    # relay status to terminal
    print("Opened port. Press reset on Central to initialize echoing. Ctrl-C to stop.")

    # execute reader and writer coroutines concurrently
    await asyncio.gather(read(reader),write(writer, action_generator))

# Reads from serial, converts all received messages to strings, and prints to terminal
async def read(reader):
    
    while True:
        line = await reader.readuntil(b'\n')
        print(str(line, 'utf-8'))

# Listen to terminal for user input and relay messages to serial
async def write(writer, action_generator):

    # this wait is not functional, it is just to give the user time to reset Central
    await asyncio.sleep(6) 

    while True:
        
        msg = await action_generator.get_action() 
        msg = msg + '\n'
        writer.write(msg.encode('UTF-8'))

def serial_process_start(action_generator):
    asyncio.run(run_connection(action_generator))

class CmdLnActionGenerator:
        def __init__(self):
            pass
        def get_action(self):
            return ainput("Message to send over serial terminal: ")

if __name__ == '__main__':
    
    serial_process_start(CmdLnActionGenerator())
#!/usr/bin/python3

import sys
import serial
import time
from multiprocessing import Process

# This script eliminates Zach's hack for IMU sensor readings, just basic I/O.

# The primary helper function here opens the serial device,
# and writes to it from raw_input.
def flush(serial_port):
    serial_port.reset_input_buffer()
    serial_port.reset_output_buffer()
    time.sleep(2)
    print("Serial port buffers flushed.")


def tx_to_serial(device_name):
    print("Running serial_tx_cmdline node with device: " + device_name)
    successful_open = False
    while not successful_open:
        try:
            serial_port = serial.Serial(port=device_name, baudrate=115200, timeout=1,
                                exclusive=False)
            successful_open = True
        except serial.serialutil.SerialException as se:
            print("Serial port not ready, received exception:")
            print(se.strerror)
            print("Retrying...")
            time.sleep(2)
    flush(serial_port)

    while True:
        try:
            to_microcontroller_msg = f'{input("Message to send over serial terminal: ")}\n'
            serial_port.write(to_microcontroller_msg.encode('UTF-8'))
        except KeyboardInterrupt:
            print("\nShutting down serial_tx_cmdline...")
            sys.exit()


"""
 The primary helper function here opens the serial device,
 and iteratively reads lines from it until stopped.
 frustratingly enough, hardware interrupts are difficult on linux,
 so we poll the device at some interval
"""


def echo_to_terminal(device_name):
    print("Running serial_rx_echo node with device: " + device_name)
    # If the device isn't ready, keep trying.
    successful_open = False
    while not successful_open:
        try:
            serial_port = serial.Serial(port=device_name, baudrate=115200, timeout=1)
            successful_open = True
        except serial.serialutil.SerialException as se:
            print("Serial port not ready, received exception:")
            print(se.strerror)
            print("Retrying...")
            time.sleep(2)
    flush(serial_port)
    print("Opened port, now echoing. Ctrl-C to stop.")

    # now = time.strftime('%d-%m-%Y_%H:%M:%S')
    # filename = f"IMU-output_{now}.csv"
    while True:
        # blocking-ly read from the port.
        # Since this is a blocking read, we CANNOT read and write at the same time...
        # BUT also need to catch keyboard interrupts.
        try:
            from_microcontroller = serial_port.readline()
            # If timed out, this call returns an empty string.
            # So, don't push anything. The string is overloaded as a boolean here.
            if from_microcontroller:
                # Check if there's a newline at the end of the string, and make sure 
                # we don't accidentally double our newlines (makes the display easier
                # to read)
                from_microcontroller = from_microcontroller.decode('UTF-8')
                if from_microcontroller[-1] == "\n":
                    # Remove the last character:
                    from_microcontroller = from_microcontroller[0:-2]
                # Echo the input back to the terminal.
                print(from_microcontroller)
                # Parse string to find if it is from the IMU and ouput to our csv
                # if from_microcontroller[0:4] == "Cent" and from_microcontroller[9] == "Q":
                #     with open(filename,"a") as f:
                #         writer = csv.writer(f,delimiter=",")
                #         writer.writerow([from_microcontroller])

        except KeyboardInterrupt:
            # Nicely shut down this script.
            print("\nShutting down serial_rx_echo...")
            sys.exit()


            # the main function: just call the helper, while parsing the serial port path.
if __name__ == '__main__':
    print("multi_serial_comms_basic.py\n 2020 Soft Machines Lab\n Basic serial I/O from UART.")
    # p2 = Process(target=echo_to_terminal, args=(sys.argv[1],))
    # so we can run in vscode more easily: check if we were given an argument, if not, hardcode
    devname = "/dev/ttyACM1"
    if len(sys.argv) > 1:
        devname = sys.argv[1]
    p2 = Process(target=echo_to_terminal, args=(devname,))
    p2.start()
    tx_to_serial(devname)

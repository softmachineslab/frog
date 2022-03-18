#!/home/zpatty/anaconda3/envs/cython-dev/bin/python

# Imports:
# because we need the command-line arguments
import sys
# and for the serial/usb/uart via pyserial:
import serial
# and for writing to csv
import csv
# and for getting a timestamp
import time

from multiprocessing import Process

# The primary helper function here opens the serial device,
# and writes to it from raw_input.

def tx_to_serial(device_name):
    # A welcome message
    print("Running serial_tx_cmdline node with device: " + device_name)
    #print(" and python version:")
    # print(sys.version)
    # Hard-code a timeout for pyserial. Seems recommended, even for tx?
    serial_timeout = 1
    # Next, do the serial setup:
    # Hard-coded: our microcontroller uses the following baud rate:
    psoc_baud = 115200
    # create the serial port object, non-exclusive (so others can use it too)
    serial_port = serial.Serial(device_name, psoc_baud, timeout=serial_timeout, exclusive=False)
    # flush out any old data
    serial_port.reset_input_buffer()
    serial_port.reset_output_buffer()
    # finishing setup.
    # print("Opened port. Ctrl-C to stop.")
    time.sleep(5)

    # If not using ROS, we'll do an infinite loop:
    while True:
        # request something to send
        try:
            to_microcontroller = input("Message to send over serial terminal: ")
            #print(to_microcontroller)
            # Concatenate a newline so that the microcontroller calls its command parser
            to_microcontroller += '\n'
            serial_port.write(to_microcontroller.encode('UTF-8'))
        except KeyboardInterrupt:
            # Nicely shut down this script.
            print("\nShutting down serial_tx_cmdline...")
            sys.exit()

            # and for writing to csv
import csv
# and for getting a timestamp
import time



# The primary helper function here opens the serial device,
# and iteratively reads lines from it until stopped.
# frustratingly enough, hardware interrupts are difficult on linux, so we poll the device at some interval

def echo_to_terminal(device_name):
    # A welcome message
    print("Running serial_rx_echo node with device: " + device_name)
    #print(" and python version:")
    # print(sys.version)
    # Hard-code a timeout for pyserial. This way, we can capture keyboard interrupts.
    # In seconds, presumably.
    serial_timeout = 1
    # Next, do the serial setup:
    # Hard-coded: our microcontroller uses the following baud rate:
    psoc_baud = 115200
    # create the serial port object
    serial_port = serial.Serial(device_name, psoc_baud, timeout=serial_timeout)
    # flush out any old data
    serial_port.reset_input_buffer()
    serial_port.reset_output_buffer()
    # finishing setup.
    print("Opened port, now echoing. Ctrl-C to stop.")

    # Get the current time for file timestamp
    now = time.strftime('%d-%m-%Y_%H:%M:%S')
    # Create filename for csv output
    filename = "IMU-output_" + now + ".csv"
    # If not using ROS, we'll do an infinite loop:
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
                if from_microcontroller[0:4] == "Cent":
                    if from_microcontroller[9]=="Q":
                        with open(filename,"a") as f:
                            writer = csv.writer(f,delimiter=",")
                            writer.writerow([from_microcontroller])

        except KeyboardInterrupt:
            # Nicely shut down this script.
            print("\nShutting down serial_rx_echo...")
            sys.exit()


            # the main function: just call the helper, while parsing the serial port path.
if __name__ == '__main__':
    try:
        # the 0-th arg is the name of the file itself, so we want the 1st.
        p2 = Process(target=echo_to_terminal, args=(sys.argv[1],))
        p2.start()
        tx_to_serial(sys.argv[1])
    except KeyboardInterrupt:
        # why is this here?
        pass
import sys
import serial
import time
from multiprocessing import Process


# The primary helper function here opens the serial device,
# and writes to it from raw_input.
def flush(serial_port):
    serial_port.reset_input_buffer()
    serial_port.reset_output_buffer()
    time.sleep(5)


def tx_to_serial(device_name):
    print("Running serial_tx_cmdline node with device: " + device_name)
    serial_port = serial.Serial(port=device_name, baudrate=115200, timeout=1,
                                exclusive=False)
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


def echo_to_terminal(device_name,sender):
    print("Running serial_rx_echo node with device: " + device_name)
    serial_port = serial.Serial(port=device_name, baudrate=115200, timeout=1, xonxoff = True)
    flush(serial_port)
    print("Opened port, now echoing. Ctrl-C to stop.")

    now = time.strftime('%d-%m-%Y_%H:%M:%S')
    filename = f"IMU-output_{now}.csv"
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
                if len(from_microcontroller) > 9:
                    if from_microcontroller[0:4] == "Cent" and from_microcontroller[9] == "Q":
                        with open(filename,"a") as f:
                            writer = csv.writer(f,delimiter=",")
                            writer.writerow([from_microcontroller])

                    if from_microcontroller[0:4] == "Cent" and from_microcontroller[9:18] == "EMERGENCY":
                        sender.send([1])
                        print("sent!")

        except KeyboardInterrupt:
            # Nicely shut down this script.
            print("\nShutting down serial_rx_echo...")
            sys.exit()


            # the main function: just call the helper, while parsing the serial port path.
if __name__ == '__main__':
    p2 = Process(target=echo_to_terminal, args=(sys.argv[1],))
    p2.start()
    tx_to_serial(sys.argv[1])

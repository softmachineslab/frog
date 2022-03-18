
import sys
import serial
import time

def flush(serial_port):
    serial_port.reset_input_buffer()
    serial_port.reset_output_buffer()
    time.sleep(0.001)

def send_command(serial_port, input_string, delay_time):
    to_microcontroller_msg = f'{input_string}\n'
    serial_port.write(to_microcontroller_msg.encode('UTF-8'))
    time.sleep(delay_time/1000)

def execute_gait_cycle(serial_port, activation_time, cooling_time):
    period = activation_time + cooling_time
    start_time = time.time()
    print(time.time() - start_time)
    send_command(serial_port, "h 0 3", activation_time)
    print(time.time() - start_time)
    send_command(serial_port, "l 0 3", period/2 - activation_time)
    print(time.time() - start_time)
    send_command(serial_port, "h 1 2", activation_time)
    print(time.time() - start_time)
    send_command(serial_port, "l 1 2", period/2 - activation_time)
    print(time.time() - start_time)

def frog_robot_run(device_name):
    # A welcome message
    print("Running serial_tx_cmdline node with device: " + device_name)
    # create the serial port object, non-exclusive (so others can use it too)
    serial_port = serial.Serial(port=device_name, baudrate=115200, timeout=1,
                                exclusive=False)    # flush out any old data
    flush(serial_port)
    # finishing setup.
    print("Opened port. Ctrl-C to stop.")
    activation_time = 80
    frequency = 3
    period = 1000/frequency # milliseconds
    cooling_time = period - activation_time

    send_command(serial_port, "p " + str(activation_time), 0)

    # If not using ROS, we'll do an infinite loop:
    while True:
        # request something to send
        try:
            execute_gait_cycle(serial_port, activation_time, cooling_time)
            
        except KeyboardInterrupt:
            # Nicely shut down this script.
            print("\nShutting down serial_tx_cmdline...")
            sys.exit()



            # the main function: just call the helper, while parsing the serial port path.
if __name__ == '__main__':
    try:
        # the 0-th arg is the name of the file itself, so we want the 1st.
        frog_robot_run(sys.argv[1])
    except KeyboardInterrupt:
        # why is this here?
        pass
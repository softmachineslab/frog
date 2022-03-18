import time
import serial
import sys 

def sleepUntil(start_time, wait_until_since_start, dt):
        # the wall time when this function should complete:
        final_time = start_time + wait_until_since_start
        # then loop until we've waited long enough
        while time.time() < final_time:
            time.sleep(dt)

def dummy(serial_port):
    
    print("now")
    timing_resolution = 0.001
    a_limb1 =  {100: "h 2 9", 300: "l 2 9", 301: "h 0 11 3 8", 600: "l 0 11 3 8", 650: "h 0 11 1 10", 950: "l 0 11 1 10", 1800: "h 9 2", 2000: "l 9 2", 2500: "s"}
    actions = {"limb1": a_limb1}
    action_name = "limb1"
    primitives = actions.keys()
    if action_name in set(primitives):
        print("Executing primitive: {} \nPlease wait...".format(action_name))
        start_time = time.time()
        command_sequence =  sorted(actions[action_name].items())
        print("Command sequence: {}".format(command_sequence))
        action = actions[action_name]
        end_ping = action[list(action)[-1]]
        #print(end_ping)
        for command in command_sequence:
            msg = command[1]
            command_time = command[0]
            #print("Attempting to send " + msg + " at time " + str(command_time) + ", current time is " + str((rospy.get_time() - start_time)*1000))
            sleepUntil(start_time,command_time/float(1000), timing_resolution)
            #print("After sleeping current time is: " + str((rospy.get_time() - start_time)*1000))

            # 1) add on the required newline
            if msg[-1] is not "\n":
                msg = msg + "\n"
            # We seem to be getting some mis-aligned commands.
            # So, before anything else, write a newline each time. This should clear the junk out?
            
            #serial_port.write("! c\n".encode('UTF-8'))
            # give the microcontroller a moment
            # maybe 20 ms?
            clear_delay = 0.02
            #time.sleep(clear_delay)
            
            #serial_port.write("! c\n".encode('UTF-8'))
            # 3) Send the message
            serial_port.write(msg.encode('UTF-8'))
            # send debugging back to terminal
            #print("Wrote command to serial port: " + msg[:-1] + " @; " + str((time.time() - start_time)*1000))


if __name__ == '__main__':
    p1 = Process(target=dummy, args=(sys.argv[1],))
    p1.start()
    tx_to_serial(sys.argv[1])
from contextlib import contextmanager
from multiprocessing import Process, Queue
from queue import Empty
import time
import cv2
import pyrealsense2 as rs
import numpy as np
import argparse
from pycpd import RigidRegistration
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
from pupil_apriltags import Detector

# update to get new location of bounding boxes
def update_tracker(frame, trackers, fps):
        # grab the new bounding box coordinates of the object
        (success, boxes) = trackers.update(frame)
        # check to see if the tracking was a success
        # loop over the bounding boxes and draw then on the frame
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # update the FPS counter
        fps.update()
        fps.stop()
        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            #("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        # loop over the info tuples and draw them on our frame
        (H, W) = frame.shape[:2]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return [frame, boxes]

# parse the input (prob get rid of this)
def parse_tracker():
        # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tracker", type=str, default="csrt",
        help="OpenCV object tracker type")
    args = vars(ap.parse_args())

    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    return [args, OPENCV_OBJECT_TRACKERS]

# Initialize tracking by specifying bounding boxes of ROI
def init_tracking(frame, trackers, april_center, OPENCV_OBJECT_TRACKERS, args):
    # Try again if we don't get enough tracked points
    num_boxes = 0
    i = 0
    while num_boxes < 2 and i <=  50:
        i = i+1
        # select region with the robot
        # bounding_box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

        # # crop frame to that region
        # cropped_frame = frame[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]), 
        #     int(bounding_box[0]):int(bounding_box[0]+bounding_box[2])]
        
        pix = np.random.normal(75, 10, 2)
        cropped_frame = frame[int(april_center[1]-pix[1]):int(april_center[1]+pix[1]), 
            int(april_center[0]-pix[0]):int(april_center[0]+pix[0])]

        cv2.imshow("Frame", cropped_frame)
        cv2.waitKey(100)
        # set number of features to find and create ORB detection object
        n = 200
        orb = cv2.ORB_create(nfeatures = n)
        # find the keypoints with ORB
        kp = orb.detect(cropped_frame,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(cropped_frame, kp)
        # convert keypoints to numpy
        points2f = cv2.KeyPoint_convert(kp)
        # set size of pixels
        w = 25
        h = w
        # convert to bounding box array format
        boxes = points2f - w
        n = np.size(boxes, 0)
        boxes = np.hstack((boxes, np.full((n,2), w)))
        # consolidate overlapping rectangles (needs to be a tuple)
        boxes = tuple(map(tuple,boxes))
        boxes, weights = cv2.groupRectangles(boxes, 1, 0.5)
        # recomute number of boxes
        num_boxes = np.size(boxes, 0)

        # convert coordinates of boxes back to original frame coordinates
        # boxes[:,0:2] = boxes[:,0:2] + np.hstack((np.full((num_boxes,1), int(bounding_box[0])), 
        #     np.full((num_boxes,1), int(bounding_box[1]))))
        boxes[:,0:2] = boxes[:,0:2] + np.hstack((np.full((num_boxes,1), int(april_center[0]-pix[0])), 
            np.full((num_boxes,1), int(april_center[1]-pix[1]))))
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
    # add a tracker object for each box (needs to be a tuple)
    boxes = tuple(map(tuple,boxes))
    
    for box in boxes:
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker, frame, box)

    if i == 50:
        boxes = None
    return boxes

# Function to get the centerpoint of boxes
def get_centerpoints(boxes):

    boxes = np.array(boxes)
    #print(boxes[:,0] + 0.5*boxes[:,2])
    centerpoints = np.array([boxes[:,0] + 0.5*boxes[:,2], boxes[:,1] + 0.5*boxes[:,3]])
    #print(centerpoints)
    return centerpoints.transpose()

# get orientation between a reference point cloud and the current point cloud using pycpd module
def get_orientation(new_centerpoints, first_centerpoints, april_pose):
    reg = RigidRegistration(**{'X': new_centerpoints, 'Y': first_centerpoints})
    reg.register()
    s, R, t = reg.get_registration_parameters()
    orientation = R.transpose()*april_pose#.transpose()
    return orientation

def get_april_pose(pipeline,cfg):

    # Create streaming profile and extract camera intrinsics
    profile = cfg.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()
    camera_params = [intr.fx, intr.fy, intr.ppx, intr.ppy]

    # Wait until detecting an apriltag before continuing on
    detections = []
    while detections == []:

        frame = get_frame(pipeline)

        # convert frame to grayscale for apriltag detection
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # initialize and detect apriltag
        detector = Detector(families='tagStandard41h12')
        detections = detector.detect(grayscale_frame, estimate_tag_pose=True, 
            camera_params = camera_params, tag_size = 0.022/8)

        cv2.imshow("Frame", grayscale_frame)
        cv2.waitKey(1)

        # calculate apriltag pose
        if len(detections) == 2:
            if detections[0].tag_id == 0:
                robot_tag = detections[0]
                goal_tag = detections[1]
            else:
                robot_tag = detections[1]
                goal_tag = detections[0]
            april_pose = np.array(robot_tag.pose_R)
            april_pose = april_pose[0:2,0:2]
            april_center = robot_tag.center
            print("Detected!")
            goal_center = goal_tag.center
            print(goal_center)
        else:
            detections = []

    return [april_pose, april_center, goal_center]


def get_frame(pipeline):
    # Wait for a coherent frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    frame = np.asanyarray(color_frame.get_data())
    return frame

# Function to execute CV and output to script
def cv_process(queue):

    # Parse user specified tracking algorithm
    args, OPENCV_OBJECT_TRACKERS = parse_tracker()

    # Initialize opencv multitracker
    trackers = cv2.MultiTracker_create()

    # Initialize the bounding box coordinates of the object we are going to track
    boxes = None

    # Set up RealSense Stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    cfg = pipeline.start(config)

    # Find apriltag orientation
    april_pose, april_center, goal_center = get_april_pose(pipeline, cfg)
    
    # Initialize the FPS throughput estimator
    fps = None

    # Initialize output
    state = []

    
    while boxes == None:
        frame = get_frame(pipeline)
        boxes = init_tracking(frame, trackers, april_center, OPENCV_OBJECT_TRACKERS, args)
    first_boxes = boxes
    new_centerpoints = get_centerpoints(first_boxes)
    first_centerpoints = new_centerpoints
    orientation = get_orientation(new_centerpoints, first_centerpoints, april_pose)
    
    fps = FPS().start()
    start_time = time.time()
    # Main Loop
    while True:

        # Ask the camera for a frame
        frame = get_frame(pipeline)
        
        # Check if the tracking has been initialized
        if boxes is not None:

            previous_boxes = boxes
            previous_centerpoints = new_centerpoints
            frame, boxes = update_tracker(frame, trackers, fps)
            this_time = time.time()
            new_boxes = boxes
            # Get new orientation
            new_centerpoints = get_centerpoints(new_boxes)
            orientation = get_orientation(new_centerpoints, first_centerpoints, april_pose)
            pt1 = np.round(first_centerpoints[0,:])
            pt2 = np.round(np.matmul(orientation, new_centerpoints[0,:].T))
            #print(pt2)
            cv2.arrowedLine(frame, tuple(pt2.astype(int)), tuple(goal_center.astype(int)), (0, 0, 255), 3, 8, 0, 0.1)

            # Collect state variables for output
            state = new_centerpoints, previous_centerpoints, this_time, start_time, orientation, april_pose, goal_center
            
        if not queue.empty():
            try:
                queue.get_nowait()
            except Empty:
                pass

        # Put state variables in queue
        queue.put(state)


        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track


        if key == ord("s"):
            continue

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            # close all windows
            cv2.destroyAllWindows()

            self.process.terminate()



if __name__ == '__main__':
    queue = Queue()
    cam_process = Process(target=cv_process, args=(queue,))
    cam_process.start()
    while True:
        # get output from the queue
        output = queue.get()
        # show the output frame
        if output != []:
            print(output[4])
        time.sleep(0.5)




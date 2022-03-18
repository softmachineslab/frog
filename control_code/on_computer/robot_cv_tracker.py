## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import argparse
from imutils.video import VideoStream
from imutils.video import FPS
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",
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

# initialize opencv multitracker
trackers = cv2.MultiTracker_create()

# initialize the bounding box coordinates of the object we are going
# to track
box = None


pipeline = rs.pipeline()
config = rs.config()
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# initialize the FPS throughput estimator
fps = None

# loop over frames from the video stream
while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    #depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # Convert images to numpy arrays
    #depth_image = np.asanyarray(depth_frame.get_data())
    frame = np.asanyarray(color_frame.get_data())
    (H, W) = frame.shape[:2]
    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame, width=600)
        # check to see if we are currently tracking an object
    if box is not None:
        # grab the new bounding box coordinates of the object
        (success, boxes) = trackers.update(frame)
        print(success)
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
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)
        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        print(type(box))
        print(box)
        trackers.add(tracker, frame, box)
        fps = FPS().start()

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

 # Stop streaming
pipeline.stop()

# close all windows
cv2.destroyAllWindows()
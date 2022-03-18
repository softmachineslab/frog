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
import random as rng
import sys

class Tracker():
    def __init__(self, queue, path_queue=None, save_filename=None, clean_queue = True, calibration = True):
        self.queue = queue
        self.path_queue = path_queue
        self.curve = []
        self.conversion = []
        self.homography = []
        self.filename = save_filename
        self.color_writer = None
        self.bw_writer = None
        self.clean_queue = clean_queue
        self.calibration = calibration
        self.tag_size = 0.047
        


        # Convert images to numpy arrays


        #camera_T = []
        #while camera_T == []:
            #camera_T= get_camera_pose(pipeline, cfg, camera_params, detector_cal)
    
    def init_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.cfg = self.pipeline.start(config)
        #device = self.cfg.get_device()
        #device.hardware_reset()
        
        # Find apriltag orientation
        self.camera_params, self.detector, self.detector_cal = self.init_april_pose()
        
        # Initialize the FPS throughput estimator
        self.fps = None

        # Initialize output
        self.state = []

        self.fps = FPS().start()

# takes in intel camera stuff and spits out calibrated parameters and a detector class
    def init_april_pose(self):

        # Create streaming profile and extract camera intrinsics
        profile = self.cfg.get_stream(rs.stream.color)
        
        intr = profile.as_video_stream_profile().get_intrinsics()
        camera_params = [intr.fx, intr.fy, intr.ppx, intr.ppy]
        # initialize apriltag
        detector_cal = Detector(families='tag36h11',
                                    nthreads=4,
                                    quad_decimate=1.0,
                                    refine_edges=True)
        detector = Detector(families='tagStandard41h12',
                                    nthreads=4,
                                    quad_decimate=1.0,
                                    refine_edges=True)

        return camera_params, detector, detector_cal

    def set_curve(self,curve):
        self.curve = curve
        

    def get_april_pose(self):
        # Wait until detecting an apriltag before continuing on
        detections = []
        while len(detections) < 1:

            frame = self.get_frame()

            # convert frame to grayscale for apriltag detection
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            tag = 47
            detections = self.detector.detect(grayscale_frame, estimate_tag_pose=True, 
                camera_params = self.camera_params, tag_size = self.tag_size)
            self.show_frame(frame)

        if self.homography == []:
            self.homography = detections[0].homography

        if self.conversion == []:
            self.corners = detections[0].corners
            diff1 = self.corners[1,:] - self.corners[0,:]
            diff2 = self.corners[3,:] - self.corners[2,:]
            self.conversion = (np.linalg.norm(diff1) + np.linalg.norm(diff2))/2/self.tag_size
            self.center = detections[0].center
            self.path_queue.put([self.conversion, self.center])

        #np.concatenate((detections[0].center, detections[0].pose_t[-1])).reshape(-1, 1)
            
        return detections[0].pose_R, detections[0].pose_t

    def get_camera_pose(self):
        frame = self.get_frame()

        # convert frame to grayscale for apriltag detection
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.detector_cal.detect(grayscale_frame, camera_params = self.camera_params)
        cv2.imshow("Frame", grayscale_frame)
        cv2.waitKey(1)

        print('Detected {} tags.\n'.format(len(detections)))

        imgPointsArr = []
        objPointsArr = []
        opointsArr = []

        board_mat, tag_size = self.get_board_centers()
        if len(detections) > 5:
            minRect = [None]*len(detections)
            coord_list = [None]*len(detections)
            rot_list = [None]*len(detections)
            for i, detection in enumerate(detections):
                imagePoints = detection.corners.reshape(1,4,2) 

                center_x = board_mat[detection.tag_id,0]
                center_y = board_mat[detection.tag_id,1]
                
                objectPoints = np.array([[-tag_size/2, -tag_size/2, 0.0],
                [ tag_size/2, -tag_size/2, 0.0],
                [ tag_size/2,  tag_size/2, 0.0],
                [-tag_size/2,  tag_size/2, 0.0]])

                objectPoints[:,0] += center_x
                objectPoints[:,1] += center_y
                cameraMatrix = np.array([[self.camera_params[0], 0, self.camera_params[2]],[0, self.camera_params[1], self.camera_params[3]], [0, 0, 1]])
                retval, rvecs, tvecs = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
                imagePoints, jacobian = cv2.projectPoints(objectPoints, rvecs, tvecs, cameraMatrix, None)
                minRect[i] = cv2.minAreaRect(np.float32(np.squeeze(imagePoints)))
                # rotated rectangle
                color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                box = cv2.boxPoints(minRect[i])
                box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
                cv2.drawContours(frame, [box], 0, color, thickness=3)
                cv2.imshow("Frame", frame)
                cv2.waitKey(100)

                rotM = cv2.Rodrigues(rvecs)[0]
                cameraPosition = np.matmul(-np.array(rotM).T, np.array(tvecs))
                coord_list[i] = cameraPosition
                rot_list[i] = rotM.T

            
            camera_t = np.mean(coord_list, axis = 0)
            camera_R = np.mean(rot_list, axis = 0)
            camera_T = self.homogeneous_transform(camera_t, camera_R)
        else:
            camera_T = []
        return camera_T
    
    def get_board_centers(self):

        tag_size = 89.89
        s = tag_size/2
        board_mat = np.array([[tag_size/2, 248.83844],
            [165.00000, 248.83844],
            [tag_size/2, 146.84945],
            [165.00000, 146.84945],
            [tag_size/2, tag_size/2],
            [165.00000, tag_size/2]])/1000
        return board_mat, tag_size/1000

    def homogeneous_transform(self, t, R):
        T = np.block([
            [R, t],
            [np.zeros((1,3)), 1]])
        return T

    def inverse_transform(self, T):
        R_inv = T[:3, :3].T
        t_inv = np.matmul(-T[:3, :3],T[:3, 3]).reshape(3,1)
        T_inv = np.block([
            [R_inv, t_inv],
            [np.zeros((1,3)), 1]])
        return T_inv

    def get_robot_pose_world(self,robot_R, robot_t, camera_T):
        robot_t = np.matmul(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), robot_t)
        robot_T = self.homogeneous_transform(robot_t, robot_R)
        robot_pose = np.matmul(camera_T,robot_T)
        #print('Robot Pose: x = {0} y = {1} z = {2}\n'.format(robot_pose[0,3], robot_pose[1,3], robot_pose[2,3]))
        return robot_pose
    
    def get_frame(self):
        # Wait for a coherent frame
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        frame = np.asanyarray(color_frame.get_data())
        return frame
    
    def show_frame(self, frame):
        if self.curve != []:
            if self.calibration == False:
                
                indices = np.rint(self.curve[:,:2]*self.conversion + self.center - self.curve[0,:2]*self.conversion).astype(int)
            else:
                num_pts = np.shape(self.curve[:,:2])[0]
                curve_t = np.hstack((self.curve[:,:2], np.zeros((num_pts,1)),np.ones((num_pts,1))))
                #camera_R = self.camera_T[:3,:3]
                #camera_T = self.camera_T
                #camera_T[:3,:3] = camera_R
                world_frame = np.matmul(self.camera_T,curve_t.T).T*1000
                world_frame[:,1] = -world_frame[:,1]
                start_pt = world_frame[0,:2]
                indices = np.rint(world_frame[:,:2] + self.center - start_pt).astype(int)
            sel1 = indices[:,0]
            sel2 = indices[:,1]
            frame[sel2,sel1] = [0,0,255]
        self.fps.stop()
        # info = [
        #         #("Tracker", args["tracker"]),
        #         ("FPS", "{:.2f}".format(self.fps.fps())),
        #     ]
        # # loop over the info tuples and draw them on our frame
        # (H, W) = frame.shape[:2]
        # for (i, (k, v)) in enumerate(info):
        #     text = "{}: {}".format(k, v)
        #     cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        #cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
        if self.color_writer is not None:
            self.color_writer.write(frame)
        cv2.imshow("Frame", frame)

        
        #cv2.resizeWindow('image', 2000,1500)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            # close all windows
            if self.color_writer is not None:
                self.color_writer.release()
            print('Exiting april_tracking.cv_process. Please kill your other processes manually, the application is now shut down.')
            print('goodbye')
            cv2.destroyAllWindows()

            self.cv_process.terminate()


    def get_euler(self,T):

        R = T[:3,:3]
        S = R[0,0]
        C = R[1,0]
        angle = np.arctan(S/C)# - np.pi/2
        if C < 0:
            angle = np.pi + angle
        elif S < 0:
            angle = 2*np.pi + angle
        return angle

    def cv_process(self):
        
        self.init_realsense()
        # Main Loop
        start_time = time.time()
        x_last = 0
        y_last = 0
        theta_last = 0
        last_time = start_time

        if self.filename is not None:
            print("Saving video to file " + self.filename)
            videospec = cv2.VideoWriter_fourcc(*'mp4v')
            color_filename = self.filename + "_color.mp4"
            bw_filename = self.filename + "_bw.mp4"
            self.color_writer = cv2.VideoWriter(color_filename, videospec, 20, (640, 480))
            self.bw_writer = cv2.VideoWriter(bw_filename, videospec, 20, (640, 480), False)
        
        if self.calibration == True:
            self.camera_T = []
            while self.camera_T == []:
                self.camera_T= self.get_camera_pose()

        while True:

            this_time = time.time()
            robot_R_C, robot_t_C = self.get_april_pose()
            if self.calibration == True:
                robot_pose = self.get_robot_pose_world(robot_R_C, robot_t_C, self.camera_T)
            else:
                robot_t = np.matmul(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), robot_t_C)
                robot_pose = np.vstack((np.hstack((robot_R_C,robot_t)),[0, 0, 0, 1]))
            x,y = robot_pose[0,3], robot_pose[1,3]

            theta = self.get_euler(robot_pose)
            #print('Robot Orientation: θ = {}\n'.format(mean_state[2]/np.pi*180))
            # Collect state variables for output
            self.state = [x, y, theta, (x-x_last)/(this_time-last_time), (y-y_last)/(this_time-last_time), (theta-theta_last)/(this_time-last_time)]

            x_last = x
            y_last = y
            theta_last = theta
            last_time = this_time
            
            if self.clean_queue == True:
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass

            if not self.path_queue.empty():
                try:
                    curve=self.path_queue.get_nowait()
                except Empty:
                    pass

                if len(curve) == 2:
                    self.path_queue.put(curve)
                elif self.calibration == False:
                    self.curve = curve
                    self.curve[:,1] = -self.curve[:,1]
                else: 
                    self.curve = curve
                    
            # Put state variables in queue
            
            self.queue.put(self.state)

            self.fps.update()
            time.sleep(0.001)
                       


if __name__ == '__main__':
    queue = Queue()
    path_queue = Queue()
    tracker = Tracker(queue, path_queue, save_filename=None, clean_queue = True, calibration = False)
    cam_process = Process(target=tracker.cv_process, args=())
    cam_process.start()
    while True:
        output = []
        # get output from the queue
        N = 3
        output = np.zeros((N,6))
        for i in range(N):
            #print('blocked on get')
            output[i,:] = queue.get()
        mean_state = np.mean(output,axis = 0)
        print('Robot velocity: x = {0} y = {1}\n'.format(mean_state[0]*1000, mean_state[1]*1000))
        print('Robot Orientation: θ = {}\n'.format(mean_state[2]/np.pi*180))
        # show the output frame
        time.sleep(0.1)








import cv2
import numpy as np
from contextlib import contextmanager
import time
import asyncio

@contextmanager
def open_camera(cam_num):
    cam = cv2.VideoCapture(cam_num)
    try:
        yield cam
    finally:
        cam.release()


async def read_frames(cam):
    ret,first_frame = cam.read()
    shape = np.shape(first_frame)
    while True:
        await asyncio.sleep(0.0000000001)
        ret,frame = cam.read()
        if ret == True:
            yield frame
        else:
            yield np.zeros(shape)


async def show_camera_feed(cam):
    async for frame in read_frames(cam):            
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
if __name__ == "__main__":
    with open_camera(0) as cam:
        asyncio.run(show_camera_feed(cam))
        
import numpy as np
import time
import cv2

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1640,
    capture_height=1232,
    framerate=30,
    flip_method=2,
):
    return (
            "nvarguscamerasrc sensor-id=%d awblock=true gainrange=\"10-10\" ispdigitalgainrange=\"2-2\" exposuretimerange=\"400 400\" aelock=true ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            capture_width,
            capture_height,
        )
    )

class CameraGilbert():
    
    def __init__(self, *args, **kwargs):
        self.cam = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0, flip_method=2), cv2.CAP_GSTREAMER)
        self.last_frametime = time.perf_counter()
    
    def enable(self, fps):
        pass
        
    def stop(self):
        self.cam.release()
        
    def getHeight(self):
        return self.cam.video_configuration.size[1]
        
    def getWidth(self):
        return self.cam.video_configuration.size[0]
    
    def getFov(self):
        return 62.2
        
    def getImage(self):
        ret_val, output = self.cam.read()
        if not ret_val:
            raise Exception("Camera read failed")
        end_new_frame = time.perf_counter()
        print(f"Frametime: {(end_new_frame - self.last_frametime)*1000:.01f}ms")
        self.last_frametime = end_new_frame
        
        return output

if __name__ == "__main__":
    cam = CameraGilbert()
    cam.enable(49)
    while True:
        cv2.imwrite("test.jpg", cam.getImage())
